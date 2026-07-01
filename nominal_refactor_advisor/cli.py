"""CLI and top-level analysis helpers.

This module contains the programmatic entrypoints used by tests and automation as
well as the command-line interface used by developers. The public helpers are the
recommended way to analyze a path or synthesize subsystem plans from findings.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Self, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta

from .analysis import (
    AnalysisPathScope,
    CachedPathAnalysisRequest,
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphAnalysisSource,
    analysis_cache_dir_for_root,
    analyze_lean_export,
    analyze_modules_with_cache,
    analyze_path,
    analyze_paths,
    plan_path,
    plan_paths,
)
from .analysis_cache import AnalysisCacheStatus, AnalysisFindingSummary
from .ast_tools import ParsedModule, PythonSourcePathPolicy, parse_python_module_roots
from .cache_paths import (
    ParseCacheDirectory,
    ParseCachePolicy,
    default_parse_cache_dir,
    semantic_descent_cache_sibling,
)
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
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
    CodemodPlanDocument,
    CodemodPlanDocumentSimulation,
    CodemodPlanJsonParser,
    CodemodPlanPreflightReport,
    CodemodPlanSequence,
    CodemodPlanSequenceSimulation,
    CodemodSelectedOperationPlanScaffoldReport,
    CodemodTargetSelector,
    CodemodSimulationReport,
    CodemodSimulationStatus,
    CodemodSourceSnapshot,
    FindingRecipePlan,
    FindingRecipeSynthesisRecord,
    FindingRecipeSynthesizer,
    JsonArray,
    JsonObject,
    JsonValue,
    NEW_SOURCE_PAYLOAD_FIELD,
    OLD_SOURCE_PAYLOAD_FIELD,
    PlannedSourceRewrite,
    RefactorRecipe,
    RefactorRecipeOperationKind,
    RefactorRecipeOperation,
    RefactorRecipeOperationPlanTemplate,
    RefactorRecipeOperationTemplate,
    RefactorRecipeRewrite,
    SourcePathCandidateAuthority,
    SourceIndexTargetSelector,
    SourceRewriteTarget,
    apply_codemod_simulation,
    codemod_class_plan_from_findings,
    codemod_plan_from_findings,
    codemod_candidates_from_impact_ranking,
    codemod_dsl_example_plan_payload,
    codemod_dsl_manifest,
    evaluate_architecture_guards,
    module_name_from_source_path,
)
from .codemod_workflow import (
    CodemodFixpointReport,
    CodemodFixpointRunner,
    CodemodFixpointScan,
    CodemodProjectedFindingReport,
    CodemodRefactorGoal,
    CodemodRefactorGoalKind,
    CodemodRefactorGoalReport,
    CodemodRefactorGoalRunner,
    CodemodSimulationFindingProjection,
    CodemodWorkflowPlan,
    CodemodWorkflowPlanJsonParser,
    codemod_workflow_plan_example_payloads,
    codemod_workflow_plan_manifests,
)
from .codemod_authoring import (
    CodemodAuthoringBundleActionRunner,
    CodemodAuthoringBundleStatusReporter,
    CodemodAuthoringWorkflowPlanner,
)
from .detectors import DetectorConfig, IssueDetector
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
from .models import RefactorFinding, RefactorPlan
from .observation_graph import build_observation_graph
from .patterns import PATTERN_SPECS, PatternId
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
    ssot_authority_findings,
)
from .semantic_descent import (
    SemanticDescentGraph,
    SemanticDescentGraphPayloadReport,
    build_finding_backed_semantic_descent_graph,
    build_semantic_descent_graph,
    load_cached_semantic_descent_graph_for_roots,
    load_latest_semantic_descent_graph_for_roots,
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
CliArgumentDefault: TypeAlias = JsonValue | Path
CliArgumentValueType: TypeAlias = type[str] | type[int] | type[float] | type[Path]
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
    default_supplied: bool = False
    dest: str | None = None
    nargs: str | int | None = None
    value_type: CliArgumentValueType | None = None

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        kwargs: dict[str, object] = {"help": self.help}
        if self.action is not None:
            kwargs["action"] = self.action
        if self.default is not None or self.default_supplied:
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
            flags=("--json-payload",),
            default="agent",
            value_type=str,
            help=(
                "JSON payload profile: agent, full, summary, or loop. The agent "
                "profile is the default and skips source index and observation "
                "graph payloads; loop emits compact findings for fast edit cycles."
            ),
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
            default=0,
            help=(
                "Number of concurrent parser workers for Python source loading. "
                "Use 0 to choose automatically."
            ),
        ),
        CliArgumentSpec(
            flags=("--analysis-workers",),
            value_type=int,
            default=0,
            help=(
                "Number of detector-analysis worker processes. Use 0 to choose "
                "automatically for package scans, or 1 for sequential analysis."
            ),
        ),
        CliArgumentSpec(
            flags=("--include-tests",),
            action="store_true",
            default=False,
            help=(
                "Include test files and test directories in source discovery. "
                "By default repo scans analyze production source only."
            ),
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
            flags=("--context-root",),
            action="append",
            dest="context_roots",
            value_type=Path,
            default=[],
            help=(
                "Parse and analyze this root for global source context while "
                "limiting reported findings to the positional paths."
            ),
        ),
        CliArgumentSpec(
            flags=("--no-auto-context-root",),
            action="store_false",
            dest="auto_context_root",
            default=True,
            help=("Do not infer package-level context roots for file-only scans."),
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
            default=None,
            default_supplied=True,
            help="Rank load-bearing refactor opportunities and dynamic trajectories.",
        ),
        CliArgumentSpec(
            flags=("--no-impact-ranking",),
            action="store_false",
            dest="include_impact_ranking",
            default=None,
            default_supplied=True,
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
                "Use '-' to read the plan from stdin. Plans enable "
                "simulatable rewrites for semantic agent-required candidates."
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
            flags=("--codemod-compose-plans",),
            value_type=Path,
            nargs="+",
            help=(
                "Load one or more codemod plan JSON documents; use '-' for one "
                "stdin document. Compose them in argument order, emit a "
                "normalized CodemodPlanDocument, and exit without scanning."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-compose-sequence",),
            value_type=Path,
            nargs="+",
            help=(
                "Load one or more codemod plan document or sequence JSON files; "
                "use '-' for one stdin document. Compose them in argument order "
                "as an ordered CodemodPlanSequence, and exit without scanning."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesize-plan",),
            action="store_true",
            help=(
                "Scan paths, synthesize executable finding-backed codemod DSL "
                "recipes, emit the synthesis report, and exit. Combine with "
                "--codemod-simulate, --codemod-diff, or --codemod-apply to "
                "execute the synthesized batch in the same scan."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesize-class-plan",),
            action="store_true",
            help=(
                "Scan paths, cluster findings into graph-derived refactor classes, "
                "and emit executable DSL plans plus evidence selectors and "
                "replacement scaffolds for each class."
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
            flags=("--codemod-plan-out",),
            value_type=Path,
            help=(
                "With a plan-producing codemod command, write the reusable "
                "CodemodPlanDocument or CodemodPlanSequence JSON to this path."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesis-authoring",),
            action="store_true",
            help=(
                "With --codemod-synthesize-plan, include opt-in authoring "
                "selectors for each synthesis record."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-bundle-out",),
            value_type=Path,
            help=(
                "With --codemod-synthesize-plan --codemod-synthesis-authoring, "
                "write per-finding selector and replacement-plan artifacts under "
                "this directory."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-status",),
            value_type=Path,
            help=(
                "Load an authoring bundle index.json, recompute workflow readiness "
                "from the current artifact files, emit JSON, and exit without "
                "scanning."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-run-action",),
            value_type=Path,
            help=(
                "Load an authoring bundle index.json, plan the command chain for "
                "--codemod-authoring-target-action, execute it, emit JSON, and "
                "exit without scanning."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-record-index",),
            value_type=int,
            default=0,
            help=(
                "Record index to use with --codemod-authoring-run-action. "
                "Defaults to 0."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-workflow-id",),
            help=(
                "Optional workflow id for --codemod-authoring-run-action. If "
                "omitted, the workflow is inferred from the target action."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-authoring-target-action",),
            help=(
                "Target action id for --codemod-authoring-run-action, such as "
                "simulate_goal_replay_plan."
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
                "against scanned paths, emit selected target rows, and exit. "
                "Use '-' to read the selector from stdin."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-target-source",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object, resolve it "
                "against scanned paths, emit exact selected target source spans, "
                "and exit. Use '-' to read the selector from stdin."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-replacement-plan",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object, resolve it "
                "against scanned paths, emit an editable replacement-source "
                "CodemodPlanDocument scaffold, and exit. Use '-' to read "
                "the selector from stdin."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-operation-plan",),
            value_type=Path,
            help=(
                "Load one codemod target selector JSON object and, with "
                "--codemod-operation-template, emit an editable "
                "apply-selected-targets CodemodPlanDocument scaffold. Use '-' "
                "to read the selector from stdin."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-node-kind",),
            action="append",
            value_type=str,
            help=(
                "Add a source-index node kind to an inline selected-operation "
                "target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-file",),
            action="append",
            value_type=str,
            help=(
                "Add a file path to an inline selected-operation source-index "
                "target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-qualname",),
            action="append",
            value_type=str,
            help=(
                "Add an exact qualname to an inline selected-operation "
                "source-index target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-file-pattern",),
            action="append",
            value_type=str,
            help=(
                "Add a file path regex to an inline selected-operation "
                "source-index target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-name-pattern",),
            action="append",
            value_type=str,
            help=(
                "Add a target-name regex to an inline selected-operation "
                "source-index target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-qualname-pattern",),
            action="append",
            value_type=str,
            help=(
                "Add a qualname regex to an inline selected-operation "
                "source-index target selector."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-operation-template",),
            value_type=Path,
            help=(
                "JSON object or array of target-local operation templates used "
                "by --codemod-selected-operation-plan. Use '-' to read "
                "templates from stdin."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-selected-replace-text",),
            nargs=2,
            help=(
                "Build a selected-target replace_text template from OLD_SOURCE "
                "and NEW_SOURCE without a separate operation-template JSON file."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-diff",),
            action="store_true",
            help="Emit a unified diff for all currently planned codemod rewrites.",
        ),
        CliArgumentSpec(
            flags=("--codemod-preflight",),
            action="store_true",
            help=(
                "Run operation-specific codemod preflight checks and emit "
                "machine-readable reports without simulating or applying rewrites."
            ),
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
            flags=("--codemod-project-findings",),
            action="store_true",
            help=(
                "With --codemod-simulate, rescan the simulated source state "
                "in memory and include before/after finding deltas."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-continuation-plan-out",),
            value_type=Path,
            help=(
                "With --codemod-project-findings, write the synthesized next-stage "
                "CodemodPlanSequence JSON to this path."
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
            flags=("--codemod-workflow-plan",),
            value_type=Path,
            help=(
                "Load a reusable codemod workflow DSL plan JSON and run it. "
                "Supported workflows: fixpoint, refactor_goal."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-fixpoint-max-iterations",),
            value_type=int,
            default=8,
            help="Maximum apply/rescan iterations for --codemod-fixpoint.",
        ),
        CliArgumentSpec(
            flags=("--codemod-fixpoint-plan-out",),
            value_type=Path,
            help=(
                "With --codemod-fixpoint, write the replayable staged "
                "CodemodPlanSequence JSON synthesized by the fixpoint run."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-refactor-goal",),
            value_type=str,
            help=(
                "Run a goal-directed staged DSL refactor. Currently supported: "
                "nominal_boundary_extraction."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-detector",),
            action="append",
            dest="codemod_goal_detectors",
            default=[],
            help=(
                "Restrict --codemod-refactor-goal, --codemod-synthesize-plan, "
                "and --codemod-synthesize-class-plan to findings from this detector "
                "(can be repeated)."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-finding-id",),
            action="append",
            dest="codemod_goal_finding_ids",
            default=[],
            help=(
                "Restrict --codemod-refactor-goal to one stable finding id "
                "(can be repeated)."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-pattern",),
            action="append",
            dest="codemod_goal_patterns",
            value_type=int,
            default=[],
            help=(
                "Restrict --codemod-refactor-goal to a canonical pattern id "
                "(can be repeated)."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-max-stages",),
            value_type=int,
            default=8,
            help="Maximum staged recipe simulations for --codemod-refactor-goal.",
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-plan-out",),
            value_type=Path,
            help=(
                "With --codemod-refactor-goal, write the replayable staged "
                "CodemodPlanSequence JSON synthesized by the goal runner."
            ),
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
class JsonFindingCounts:
    """Compact finding-count projection for tight-loop JSON payloads."""

    summary: AnalysisFindingSummary

    def to_dict(self) -> JsonObject:
        return {
            "by_pattern": tuple(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": PATTERN_SPECS[PatternId(pattern_id)].name,
                    "count": count,
                }
                for pattern_id, count in (
                    (item.pattern_id, item.count)
                    for item in self.summary.pattern_counts
                )
            ),
            "by_detector": tuple(
                {"detector_id": detector_id, "count": count}
                for detector_id, count in (
                    (item.detector_id, item.count)
                    for item in self.summary.detector_counts
                )
            ),
        }


class JsonFindingPayloadMode(Enum):
    """Finding-detail level emitted by one JSON payload profile."""

    full = "full"
    counts_only = "counts_only"
    semantic_work_queue = "semantic_work_queue"


class JsonFindingPayloadProjection:
    """Build the JSON finding list for one payload mode."""

    @classmethod
    def payload(
        cls,
        findings: list[RefactorFinding],
        mode: JsonFindingPayloadMode,
    ) -> list[JsonObject]:
        if mode is JsonFindingPayloadMode.full:
            return [finding.to_dict() for finding in findings]
        return []


@dataclass(frozen=True)
class JsonPayloadSections:
    """Declared section policy for one JSON payload profile."""

    finding_payload_mode: JsonFindingPayloadMode = JsonFindingPayloadMode.full
    source_index: bool = True
    observation_graph: bool = True
    observation_fibers: bool = True
    semantic_descent_graph: bool = True
    semantic_refactor_gate: bool = True
    candidate_payload: bool = True
    finding_recipe_plan: bool = True
    payload_timing: bool = False
    default_impact_ranking: bool = True

    @property
    def needs_observation_graph(self) -> bool:
        return self.observation_graph or self.observation_fibers

    @property
    def needs_candidate_projection(self) -> bool:
        return self.semantic_refactor_gate or self.candidate_payload

    @property
    def lightweight_status_payload(self) -> bool:
        return (
            not self.source_index
            and not self.needs_observation_graph
            and not self.semantic_descent_graph
            and not self.semantic_refactor_gate
            and not self.candidate_payload
            and not self.finding_recipe_plan
            and not self.default_impact_ranking
        )


@dataclass(frozen=True)
class JsonPayloadSourceSnapshotDemand:
    """Source-snapshot demand induced by one payload section policy."""

    sections: JsonPayloadSections
    impact_ranking_report: RefactorImpactRankingReport | None
    candidate_selection: CodemodCandidateSelection

    @property
    def needs_generated_candidate_selection(self) -> bool:
        needs_generated_candidates = (
            self.impact_ranking_report is not None
            and self.candidate_selection is None
            and self.sections.needs_candidate_projection
        )
        return needs_generated_candidates

    @property
    def needs_source_snapshot(self) -> bool:
        return (
            self.sections.source_index
            or self.sections.finding_recipe_plan
            or self.needs_generated_candidate_selection
        )


class JsonPayloadProfile(Enum):
    """Named JSON payload profiles for CLI and programmatic callers."""

    full = JsonPayloadSections()
    agent = JsonPayloadSections(
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_impact_ranking=False,
    )
    summary = JsonPayloadSections(
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        semantic_descent_graph=False,
        semantic_refactor_gate=False,
        candidate_payload=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_impact_ranking=False,
    )
    loop = JsonPayloadSections(
        finding_payload_mode=JsonFindingPayloadMode.counts_only,
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        semantic_descent_graph=False,
        semantic_refactor_gate=False,
        candidate_payload=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_impact_ranking=False,
    )

    @classmethod
    def from_cli_value(cls, raw_value: str) -> "JsonPayloadProfile":
        try:
            return cls[raw_value]
        except KeyError as error:
            choices = ", ".join(profile.name for profile in cls)
            raise ValueError(
                f"unknown JSON payload profile {raw_value!r}; choose one of {choices}"
            ) from error

    @property
    def sections(self) -> JsonPayloadSections:
        return self.value


@dataclass(frozen=True)
class JsonPayloadImpactRankingPolicy:
    """Resolved impact-ranking policy for one CLI JSON payload profile."""

    explicit_request: bool | None
    json_enabled: bool
    payload_profile: JsonPayloadProfile

    @property
    def include_impact_ranking(self) -> bool:
        if self.explicit_request is not None:
            return self.explicit_request
        if self.json_enabled:
            return self.payload_profile.sections.default_impact_ranking
        return True

    @property
    def lightweight_profile_acknowledges_raw_findings(self) -> bool:
        return (
            self.json_enabled
            and self.payload_profile.sections.lightweight_status_payload
            and not self.include_impact_ranking
        )


class JsonPreparseCachePayloadMode(Enum):
    """Pre-parse cache payload mode for JSON scans."""

    DISABLED = ("disabled", False, False)
    LOOP_SUMMARY = ("loop_summary", True, False)
    SEMANTIC_GRAPH_PAYLOAD = ("semantic_graph_payload", False, True)

    def __new__(
        cls,
        value: str,
        evidence_local_partial: bool,
        focused_evidence_local_partial: bool,
    ) -> "JsonPreparseCachePayloadMode":
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __init__(
        self,
        value: str,
        evidence_local_partial: bool,
        focused_evidence_local_partial: bool,
    ) -> None:
        del value
        self._evidence_local_partial = evidence_local_partial
        self._focused_evidence_local_partial = focused_evidence_local_partial

    @property
    def enabled(self) -> bool:
        return self is not type(self).DISABLED

    @property
    def requires_semantic_descent_cache(self) -> bool:
        return self is type(self).SEMANTIC_GRAPH_PAYLOAD

    def reuse_policy(self, *, focused_report_filter: bool) -> FastCacheReusePolicy:
        evidence_local_partial = self._evidence_local_partial or (
            focused_report_filter and self._focused_evidence_local_partial
        )
        if evidence_local_partial:
            return FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL
        return FastCacheReusePolicy.EXACT_ONLY


@dataclass(frozen=True)
class JsonSummaryPreparseCachePolicy:
    """Decide whether lightweight JSON can consult cache before parsing."""

    json_enabled: bool
    payload_profile: JsonPayloadProfile
    load_bearing_ranking_enabled: bool
    parsed_modules_required: bool
    analysis_cache_dir: Path | None
    focused_report_filter: bool = False

    @property
    def cache_lookup_enabled(self) -> bool:
        return (
            self.json_enabled
            and not self.load_bearing_ranking_enabled
            and not self.parsed_modules_required
            and self.analysis_cache_dir is not None
        )

    @property
    def enabled(self) -> bool:
        return self.mode.enabled

    @property
    def mode(self) -> JsonPreparseCachePayloadMode:
        if not self.cache_lookup_enabled:
            return JsonPreparseCachePayloadMode.DISABLED
        sections = self.payload_profile.sections
        if sections.lightweight_status_payload:
            return JsonPreparseCachePayloadMode.LOOP_SUMMARY
        if (
            sections.semantic_descent_graph
            and not sections.source_index
            and not sections.needs_observation_graph
            and not sections.finding_recipe_plan
        ):
            return JsonPreparseCachePayloadMode.SEMANTIC_GRAPH_PAYLOAD
        return JsonPreparseCachePayloadMode.DISABLED

    @property
    def uses_evidence_local_partial_reuse(self) -> bool:
        return (
            self.mode.reuse_policy(focused_report_filter=self.focused_report_filter)
            is FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL
        )


@dataclass(frozen=True)
class FastPreparseSemanticDescentContext:
    """Semantic-descent graph context available to a pre-parse cache lookup."""

    analysis_source: SemanticDescentGraphAnalysisSource
    latest_graph: SemanticDescentGraph | None = None


@dataclass(frozen=True, kw_only=True)
class FastPreparseSemanticDescentSourceAuthority:
    """Load cached repo graph context for evidence-local partial scans."""

    preparse_cache_policy: JsonSummaryPreparseCachePolicy
    base_source: SemanticDescentGraphAnalysisSource
    roots: tuple[Path, ...]
    semantic_descent_cache_dir: Path | None
    source_policy: PythonSourcePathPolicy
    use_cache: bool

    def context(self) -> FastPreparseSemanticDescentContext:
        latest_graph = self.latest_graph()
        if latest_graph is None:
            return FastPreparseSemanticDescentContext(self.base_source)
        return FastPreparseSemanticDescentContext(
            SemanticDescentGraphAnalysisSource(
                cached_graph=latest_graph,
                cache_dir=self.semantic_descent_cache_dir,
                cache_roots=self.roots,
                source_policy=self.source_policy,
                use_cache=self.use_cache,
            ),
            latest_graph=latest_graph,
        )

    def latest_graph(self) -> SemanticDescentGraph | None:
        if not self.preparse_cache_policy.uses_evidence_local_partial_reuse:
            return None
        return load_latest_semantic_descent_graph_for_roots(
            self.roots,
            cache_dir=self.semantic_descent_cache_dir,
            source_policy=self.source_policy,
        )


@dataclass(frozen=True)
class JsonPayloadBuildTiming:
    """Wall-clock time spent in optional JSON payload sections."""

    observation_graph_seconds: float = 0.0
    semantic_descent_graph_seconds: float = 0.0
    source_snapshot_seconds: float = 0.0
    source_index_payload_seconds: float = 0.0
    semantic_refactor_gate_seconds: float = 0.0
    finding_recipe_plan_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> JsonObject:
        return {
            "observation_graph_seconds": self.observation_graph_seconds,
            "semantic_descent_graph_seconds": self.semantic_descent_graph_seconds,
            "source_snapshot_seconds": self.source_snapshot_seconds,
            "source_index_payload_seconds": self.source_index_payload_seconds,
            "semantic_refactor_gate_seconds": self.semantic_refactor_gate_seconds,
            "finding_recipe_plan_seconds": self.finding_recipe_plan_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class JsonFindingPayloadEnvelope:
    """Shared finding/planning envelope for JSON scan payloads."""

    summary: AnalysisFindingSummary
    section_policy: JsonPayloadSections
    finding_payload: list[JsonObject]
    plan_payload: tuple[JsonObject, ...] = ()

    def to_dict(self) -> JsonObject:
        return {
            "findings": self.finding_payload,
            "plans": self.plan_payload,
            "finding_payload_mode": (self.section_policy.finding_payload_mode.value),
            "finding_count": self.summary.finding_count,
            "finding_counts": JsonFindingCounts(self.summary).to_dict(),
        }


@dataclass(frozen=True)
class JsonLoopCachePayloadBuilder:
    """Build loop JSON directly from an exact cache-summary hit."""

    summary: AnalysisFindingSummary
    timing: ScanTiming

    def to_dict(self) -> JsonObject:
        payload_started = perf_counter()
        payload = JsonFindingPayloadEnvelope(
            summary=self.summary,
            section_policy=JsonPayloadProfile.loop.sections,
            finding_payload=[],
        ).to_dict()
        payload["timing"] = self.timing.to_dict()
        payload["payload_timing"] = JsonPayloadBuildTiming(
            total_seconds=round(perf_counter() - payload_started, 3),
        ).to_dict()
        return payload


@dataclass(frozen=True, kw_only=True)
class SemanticDescentCacheDirAuthority(ParseCacheDirectory):
    """Resolve the semantic-descent graph cache from the parse-cache authority."""

    def cache_dir(self) -> Path | None:
        if not self.use_parse_cache or self.parse_cache_dir is None:
            return None
        return semantic_descent_cache_sibling(self.parse_cache_dir)

    def graph_for_modules(self, modules: list[ParsedModule]) -> SemanticDescentGraph:
        return build_semantic_descent_graph(
            modules,
            cache_dir=self.cache_dir(),
            use_cache=self.use_parse_cache,
        )


@dataclass(frozen=True, kw_only=True)
class JsonSemanticDescentPayloadSource(SemanticDescentCacheDirAuthority):
    """Repository graph source for the semantic-descent JSON section."""

    modules: list[ParsedModule]
    cached_repository_graph: SemanticDescentGraph | None = None

    @property
    def available(self) -> bool:
        return bool(self.modules) or self.cached_repository_graph is not None

    def repository_graph(self) -> SemanticDescentGraph:
        if self.cached_repository_graph is not None:
            return self.cached_repository_graph
        return self.graph_for_modules(self.modules)


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
    semantic_descent_source: JsonSemanticDescentPayloadSource | None = None
    payload_sections: JsonPayloadSections = JsonPayloadProfile.full.sections
    raw_findings: bool = False

    def to_dict(self) -> JsonObject:
        payload_started = perf_counter()
        sections = self.payload_sections
        finding_tuple = tuple(self.findings)
        payload = JsonFindingPayloadEnvelope(
            summary=AnalysisFindingSummary.from_findings(finding_tuple),
            section_policy=sections,
            finding_payload=JsonFindingPayloadProjection.payload(
                self.findings,
                sections.finding_payload_mode,
            ),
            plan_payload=tuple(plan.to_dict() for plan in self.plans),
        ).to_dict()
        snapshot_demand = JsonPayloadSourceSnapshotDemand(
            sections=sections,
            impact_ranking_report=self.impact_ranking,
            candidate_selection=self.codemod_candidates,
        )
        observation_graph_seconds = 0.0
        if sections.needs_observation_graph:
            started = perf_counter()
            graph = build_observation_graph(self.modules)
            observation_graph_seconds = round(perf_counter() - started, 3)
            if sections.observation_graph:
                payload["observations"] = [asdict(item) for item in graph.observations]
            if sections.observation_fibers:
                payload["fibers"] = [asdict(item) for item in graph.fibers]
        semantic_descent_graph_seconds = 0.0
        semantic_descent_source = (
            self.semantic_descent_source
            if self.semantic_descent_source is not None
            else JsonSemanticDescentPayloadSource(modules=self.modules)
        )
        if (
            sections.semantic_descent_graph
            and finding_tuple
            and semantic_descent_source.available
        ):
            started = perf_counter()
            payload["semantic_descent_graph"] = (
                SemanticDescentGraphPayloadReport.from_graphs(
                    semantic_descent_source.repository_graph(),
                    finding_backed_graph=build_finding_backed_semantic_descent_graph(
                        ssot_authority_findings(finding_tuple),
                        semantic_mirror_detector_ids=(
                            IssueDetector.semantic_mirror_detector_ids()
                        ),
                        authority_evidence_index_by_detector_id=(
                            IssueDetector.semantic_mirror_authority_evidence_indices()
                        ),
                    ),
                ).to_dict()
            )
            semantic_descent_graph_seconds = round(perf_counter() - started, 3)
        source_snapshot = self.source_snapshot
        built_source_index_seconds = 0.0
        if source_snapshot is None and snapshot_demand.needs_source_snapshot:
            started = perf_counter()
            source_snapshot = CodemodSourceSnapshot.from_modules(
                self.modules,
                self.findings,
            )
            built_source_index_seconds = round(perf_counter() - started, 3)
        source_index_payload_seconds = 0.0
        if sections.source_index and source_snapshot is not None:
            started = perf_counter()
            payload["source_index"] = source_snapshot.source_index.to_dict()
            source_index_payload_seconds = round(perf_counter() - started, 3)
        timing = self.timing
        if timing is not None and built_source_index_seconds:
            timing = ScanTiming(
                parse_seconds=timing.parse_seconds,
                analysis_seconds=timing.analysis_seconds,
                planning_seconds=timing.planning_seconds,
                source_index_seconds=built_source_index_seconds,
                analysis_cache_status=timing.analysis_cache_status,
            )
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
            if (
                codemod_candidates is None
                and source_snapshot is not None
                and sections.needs_candidate_projection
            ):
                source_index = source_snapshot.source_index
                codemod_candidates = codemod_candidates_from_impact_ranking(
                    self.impact_ranking,
                    source_index,
                )
                codemod_candidates = source_snapshot.candidates_with_automated_rewrites(
                    codemod_candidates,
                )
        semantic_refactor_gate_seconds = 0.0
        semantic_gate_report = SemanticRefactorGateReport.inactive()
        if sections.semantic_refactor_gate:
            started = perf_counter()
            semantic_gate_report = SemanticRefactorGateReport.from_optional_scan(
                codemod_candidates,
                impact_ranking=self.impact_ranking,
                findings=tuple(self.findings),
            )
            payload["semantic_refactor_gate"] = semantic_gate_report.to_dict()
            if semantic_gate_report.active:
                payload["findings"] = semantic_gate_report.finding_payload()
                payload["finding_payload_mode"] = (
                    JsonFindingPayloadMode.semantic_work_queue.value
                )
                payload["active_finding_surface"] = "semantic_refactor_work_queue"
                payload["raw_findings_default"] = (
                    semantic_gate_report.raw_findings_default
                )
                payload["supporting_raw_finding_count"] = len(self.findings)
                if self.raw_findings:
                    payload["supporting_raw_findings"] = [
                        finding.to_dict() for finding in self.findings
                    ]
            semantic_refactor_gate_seconds = round(perf_counter() - started, 3)
        if not semantic_gate_report.active:
            payload["active_finding_surface"] = "raw_findings"
        if sections.candidate_payload and codemod_candidates is not None:
            payload["codemod_candidates"] = tuple(
                candidate.to_dict() for candidate in codemod_candidates
            )
        finding_recipe_plan_seconds = 0.0
        if sections.finding_recipe_plan and source_snapshot is not None:
            started = perf_counter()
            payload["finding_recipe_plan"] = source_snapshot.plan_from_findings(
                self.findings,
            ).to_dict()
            finding_recipe_plan_seconds = round(perf_counter() - started, 3)
        if self.scan_guard_report is not None:
            payload["architecture_guard_report"] = self.scan_guard_report.to_dict()
        if sections.payload_timing:
            payload["payload_timing"] = JsonPayloadBuildTiming(
                observation_graph_seconds=observation_graph_seconds,
                semantic_descent_graph_seconds=semantic_descent_graph_seconds,
                source_snapshot_seconds=built_source_index_seconds,
                source_index_payload_seconds=source_index_payload_seconds,
                semantic_refactor_gate_seconds=semantic_refactor_gate_seconds,
                finding_recipe_plan_seconds=finding_recipe_plan_seconds,
                total_seconds=round(perf_counter() - payload_started, 3),
            ).to_dict()
        return payload


STDIN_JSON_DOCUMENT_TOKEN = "-"


@dataclass(frozen=True)
class JsonDocumentSource:
    """CLI JSON source backed by one path or the stdin token."""

    path: Path

    @property
    def reads_stdin(self) -> bool:
        return self.path.as_posix() == STDIN_JSON_DOCUMENT_TOKEN

    def load(self) -> JsonValue:
        if self.reads_stdin:
            return cast(JsonValue, json.loads(sys.stdin.read()))
        return cast(JsonValue, json.loads(self.path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class JsonDocumentInput:
    """One user-facing CLI option that may consume a JSON document."""

    option_name: str
    path: Path

    @property
    def reads_stdin(self) -> bool:
        return JsonDocumentSource(self.path).reads_stdin


@dataclass(frozen=True)
class JsonDocumentInputSet:
    """Validate stdin document use across one CLI invocation."""

    inputs: tuple[JsonDocumentInput, ...]

    @classmethod
    def from_option_paths(
        cls,
        option_paths: tuple[tuple[str, tuple[Path | None, ...]], ...],
    ) -> "JsonDocumentInputSet":
        return cls(
            tuple(
                JsonDocumentInput(option_name, path)
                for option_name, paths in option_paths
                for path in paths
                if path is not None
            )
        )

    @property
    def stdin_inputs(self) -> tuple[JsonDocumentInput, ...]:
        return tuple(item for item in self.inputs if item.reads_stdin)

    def require_at_most_one_stdin(self, parser: argparse.ArgumentParser) -> None:
        if len(self.stdin_inputs) <= 1:
            return
        option_names = ", ".join(item.option_name for item in self.stdin_inputs)
        parser.error(
            "stdin JSON document token '-' can be used by only one codemod "
            f"document option per invocation: {option_names}"
        )


def load_authority_boundary_plans(path: Path) -> tuple[AuthorityBoundaryPlan, ...]:
    """Load caller-supplied authority boundary plans from JSON."""

    return load_codemod_plan_sequence(path).authority_boundaries


def load_codemod_plan_document(path: Path) -> CodemodPlanDocument:
    """Load caller-supplied codemod rewrites and guard invariants from JSON."""

    payload = cast(
        JsonObject | JsonArray,
        JsonDocumentSource(path).load(),
    )
    return CodemodPlanDocument.from_json_value(payload)


def load_codemod_plan_sequence(path: Path) -> CodemodPlanSequence:
    """Load one codemod document or staged codemod sequence from JSON."""

    payload = cast(
        JsonObject | JsonArray,
        JsonDocumentSource(path).load(),
    )
    return CodemodPlanJsonParser().parse_sequence(payload)


def load_codemod_workflow_plan(path: Path) -> CodemodWorkflowPlan:
    """Load one reusable codemod workflow plan from JSON."""

    payload = cast(
        JsonObject,
        JsonDocumentSource(path).load(),
    )
    return CodemodWorkflowPlanJsonParser().parse_plan(payload)


def load_codemod_plan_validation_payload(path: Path) -> JsonObject:
    """Load a codemod document or sequence and return its normalized JSON shape."""

    payload = cast(
        JsonObject | JsonArray,
        JsonDocumentSource(path).load(),
    )
    parser = CodemodPlanJsonParser()
    if isinstance(payload, dict) and parser.stages_field in payload:
        return parser.parse_sequence(payload).to_dict()
    return parser.parse_document(payload).to_dict()


def load_codemod_target_selector(path: Path) -> CodemodTargetSelector:
    """Load one registry-backed codemod target selector from JSON."""

    payload = JsonDocumentSource(path).load()
    if not isinstance(payload, Mapping):
        raise ValueError("codemod selector JSON must be an object")
    return CodemodTargetSelector.from_dict(cast(Mapping[str, JsonValue], payload))


def cli_string_tuple(
    value: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Normalize an optional argparse string/list value into a tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(cast(list[str], value))
    if isinstance(value, tuple):
        return value
    raise TypeError(
        f"Expected CLI string value or sequence, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class CodemodSelectedOperationSourceManifest:
    """Manifest row for one selected-operation selector/template source."""

    source_id: str
    source_label: str
    option_names: tuple[str, ...]
    class_name: str
    registry_order: int

    def to_dict(self) -> JsonObject:
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "option_names": self.option_names,
            "class_name": self.class_name,
            "registry_order": self.registry_order,
        }


class SelectedOperationCliSource(ABC):
    """Shared contract for selected-operation CLI source registries."""

    source_id: ClassVar[str]
    source_family_label: ClassVar[str]
    source_label: ClassVar[str]
    option_names: ClassVar[tuple[str, ...]]
    registry_order: ClassVar[int]

    @classmethod
    def ordered_source_types(cls) -> tuple[type, ...]:
        return tuple(
            sorted(
                cls.__registry__.values(),
                key=lambda source_type: source_type.registry_order,
            )
        )

    @classmethod
    def selected_sources(
        cls,
        args: argparse.Namespace,
    ) -> tuple["SelectedOperationCliSource", ...]:
        return tuple(
            source
            for source in (source_type() for source_type in cls.ordered_source_types())
            if source.is_supplied(args)
        )

    @classmethod
    def selected_source_labels(cls, args: argparse.Namespace) -> tuple[str, ...]:
        return tuple(source.source_label for source in cls.selected_sources(args))

    @classmethod
    def required_source(
        cls,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> Self:
        sources = cls.selected_sources(args)
        if len(sources) == 1:
            return cast(Self, sources[0])
        if not sources:
            parser.error(f"{cls.source_family_label} requires one source")
        parser.error(
            f"{cls.source_family_label} sources are mutually exclusive: "
            + ", ".join(source.source_label for source in sources)
        )

    def source_manifest(self) -> CodemodSelectedOperationSourceManifest:
        return CodemodSelectedOperationSourceManifest(
            source_id=self.source_id,
            source_label=self.source_label,
            option_names=self.option_names,
            class_name=type(self).__name__,
            registry_order=self.registry_order,
        )

    @abstractmethod
    def is_supplied(self, args: argparse.Namespace) -> bool:
        raise NotImplementedError


class SelectedOperationTargetSelectorSource(
    SelectedOperationCliSource,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered CLI source for selected-operation target selectors."""

    __registry__: ClassVar[dict[str, type["SelectedOperationTargetSelectorSource"]]] = (
        {}
    )
    __registry_key__ = "source_id"
    __skip_if_no_key__ = True
    source_family_label = "selected-operation target selector"

    @abstractmethod
    def target_selector(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> CodemodTargetSelector:
        raise NotImplementedError


class JsonSelectedOperationTargetSelectorSource(SelectedOperationTargetSelectorSource):
    """Load a selected-operation target selector from JSON."""

    source_id = "json_target_selector"
    source_label = "--codemod-selected-operation-plan"
    option_names = ("--codemod-selected-operation-plan",)
    registry_order = 10

    def is_supplied(self, args: argparse.Namespace) -> bool:
        return args.codemod_selected_operation_plan is not None

    def target_selector(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> CodemodTargetSelector:
        try:
            return load_codemod_target_selector(args.codemod_selected_operation_plan)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))


SelectedOperationSelectorValueReader: TypeAlias = Callable[
    [argparse.Namespace],
    tuple[str, ...],
]


@dataclass(frozen=True)
class InlineSourceIndexSelectorCliField:
    """Projection from one CLI flag to one source_index_target payload field."""

    option_name: str
    payload_field_name: str
    value_reader: SelectedOperationSelectorValueReader

    def payload_item(self, args: argparse.Namespace) -> tuple[str, tuple[str, ...]]:
        return self.payload_field_name, self.value_reader(args)


@dataclass(frozen=True)
class CliOptionStringTupleValueReader:
    option_name: str

    @property
    def argparse_dest_name(self) -> str:
        return self.option_name.removeprefix("--").replace("-", "_")

    def __call__(self, args: argparse.Namespace) -> tuple[str, ...]:
        return cli_string_tuple(vars(args)[self.argparse_dest_name])


class InlineSourceIndexSelectedOperationTargetSelectorSource(
    SelectedOperationTargetSelectorSource
):
    """Build a source_index_target selector from repeatable CLI operands."""

    source_id = "inline_source_index_target"
    source_label = "inline source_index_target selector"
    option_names = (
        "--codemod-selected-node-kind",
        "--codemod-selected-file",
        "--codemod-selected-qualname",
        "--codemod-selected-file-pattern",
        "--codemod-selected-name-pattern",
        "--codemod-selected-qualname-pattern",
    )
    registry_order = 20
    cli_option_names: ClassVar[tuple[str, ...]] = (
        "--codemod-selected-node-kind",
        "--codemod-selected-file",
        "--codemod-selected-qualname",
        "--codemod-selected-file-pattern",
        "--codemod-selected-name-pattern",
        "--codemod-selected-qualname-pattern",
    )

    @classmethod
    def cli_fields(cls) -> tuple[InlineSourceIndexSelectorCliField, ...]:
        return tuple(
            InlineSourceIndexSelectorCliField(
                option_name=option_name,
                payload_field_name=selector_binding.field_name,
                value_reader=CliOptionStringTupleValueReader(option_name),
            )
            for option_name, selector_binding in zip(
                cls.cli_option_names,
                SourceIndexTargetSelector.selector_payload_bindings,
                strict=True,
            )
        )

    def is_supplied(self, args: argparse.Namespace) -> bool:
        return any(values for _, values in self.payload_items(args))

    def target_selector(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> CodemodTargetSelector:
        del parser
        payload_fields = {
            field_name: values
            for field_name, values in self.payload_items(args)
            if values
        }
        payload = {
            "selector": SourceIndexTargetSelector().to_dict()["selector"],
            **payload_fields,
        }
        return CodemodTargetSelector.from_dict(payload)

    def payload_items(
        self,
        args: argparse.Namespace,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(field.payload_item(args) for field in self.cli_fields())


def load_codemod_operation_templates(
    path: Path,
) -> tuple[RefactorRecipeOperationTemplate, ...]:
    """Load one or more target-local codemod operation templates from JSON."""

    payload = JsonDocumentSource(path).load()
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


def load_codemod_operation_plan_template(
    path: Path,
) -> RefactorRecipeOperationPlanTemplate:
    """Load a selected-target operation plan template from JSON."""

    payload = JsonDocumentSource(path).load()
    return RefactorRecipeOperationPlanTemplate.from_json_value(payload)


class SelectedOperationTemplateSource(
    SelectedOperationCliSource,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered CLI source for selected-target operation templates."""

    __registry__: ClassVar[dict[str, type["SelectedOperationTemplateSource"]]] = {}
    __registry_key__ = "source_id"
    __skip_if_no_key__ = True
    source_family_label = "selected-operation template"

    @abstractmethod
    def operation_plan_template(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> RefactorRecipeOperationPlanTemplate:
        raise NotImplementedError


class JsonSelectedOperationTemplateSource(SelectedOperationTemplateSource):
    """Load selected-target operation templates from JSON."""

    source_id = "json_operation_template"
    source_label = "--codemod-operation-template"
    option_names = ("--codemod-operation-template",)
    registry_order = 10

    def is_supplied(self, args: argparse.Namespace) -> bool:
        return args.codemod_operation_template is not None

    def operation_plan_template(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> RefactorRecipeOperationPlanTemplate:
        try:
            return load_codemod_operation_plan_template(args.codemod_operation_template)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))


class ReplaceTextSelectedOperationTemplateSource(SelectedOperationTemplateSource):
    """Build one replace_text template directly from CLI operands."""

    source_id = "replace_text_operands"
    source_label = "--codemod-selected-replace-text"
    option_names = ("--codemod-selected-replace-text",)
    registry_order = 20

    def is_supplied(self, args: argparse.Namespace) -> bool:
        return args.codemod_selected_replace_text is not None

    def operation_plan_template(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> RefactorRecipeOperationPlanTemplate:
        del parser
        old_source, new_source = cast(
            tuple[str, str],
            tuple(args.codemod_selected_replace_text),
        )
        return RefactorRecipeOperationPlanTemplate.from_operation_templates(
            (
                RefactorRecipeOperationTemplate.from_payload(
                    {
                        "operation": RefactorRecipeOperationKind.REPLACE_TEXT.value,
                        OLD_SOURCE_PAYLOAD_FIELD: old_source,
                        NEW_SOURCE_PAYLOAD_FIELD: new_source,
                    }
                ),
            )
        )


def write_cli_json_artifact(path: Path | None, payload: JsonObject) -> None:
    """Write a machine-readable CLI artifact when the caller requested one."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def codemod_plan_output_supported(args: argparse.Namespace) -> bool:
    """Return whether the selected command can produce reusable plan JSON."""

    return any(
        (
            args.codemod_dsl_example_plan,
            args.codemod_validate_plan,
            args.codemod_compose_plans is not None,
            args.codemod_compose_sequence is not None,
            args.codemod_synthesize_plan,
            args.codemod_synthesize_class_plan,
            args.codemod_replacement_plan is not None,
            bool(SelectedOperationTargetSelectorSource.selected_sources(args)),
            args.codemod_fixpoint,
            args.codemod_refactor_goal is not None,
            args.codemod_workflow_plan is not None,
        )
    )


def cli_artifact_slug(value: str) -> str:
    """Convert a report identifier into a stable filesystem-friendly stem."""

    if not value:
        raise ValueError("CLI artifact slug source cannot be empty")
    characters: list[str] = []
    for character in value:
        if character.isalnum() or character in ("-", "_"):
            characters.append(character)
            continue
        characters.append("_")
    cleaned = "".join(characters).strip("_")
    if not cleaned:
        raise ValueError("CLI artifact slug source has no filesystem-safe characters")
    return cleaned


class CodemodAuthoringCommandActionId(str, Enum):
    """Stable action ids for replayable codemod authoring commands."""

    RESOLVE_SELECTOR = "resolve_selector"
    SCAFFOLD_REPLACEMENT_PLAN = "scaffold_replacement_plan"
    VALIDATE_REPLACEMENT_PLAN = "validate_replacement_plan"
    SIMULATE_REPLACEMENT_PLAN = "simulate_replacement_plan"
    APPLY_REPLACEMENT_PLAN = "apply_replacement_plan"
    SCAFFOLD_SELECTED_OPERATION_PLAN = "scaffold_selected_operation_plan"
    PREFLIGHT_SELECTED_OPERATION_PLAN = "preflight_selected_operation_plan"
    SIMULATE_SELECTED_OPERATION_PLAN = "simulate_selected_operation_plan"
    APPLY_SELECTED_OPERATION_PLAN = "apply_selected_operation_plan"
    RUN_GOAL_REFACTOR = "run_goal_refactor"
    SIMULATE_GOAL_REPLAY_PLAN = "simulate_goal_replay_plan"
    APPLY_GOAL_REPLAY_PLAN = "apply_goal_replay_plan"


class CodemodAuthoringWorkflowId(str, Enum):
    """Stable workflow ids for codemod authoring bundles."""

    REPLACEMENT_PLAN = "replacement_plan"
    SELECTED_OPERATION_TEMPLATE = "selected_operation_template"
    GOAL_REFACTOR = "goal_refactor"


class CodemodAuthoringBundleControlActionId(str, Enum):
    """Stable action ids for authoring bundle control commands."""

    AUTHORING_STATUS = "authoring_status"
    AUTHORING_RUN_ACTION = "authoring_run_action"


class CodemodAuthoringArtifactRole(str, Enum):
    """Stable artifact roles used by codemod authoring workflows."""

    EVIDENCE_SELECTOR_FILE = "evidence_selector"
    REPLACEMENT_SCAFFOLD_FILE = "replacement_scaffold"
    REPLACEMENT_PLAN_FILE = "replacement_plan"
    SELECTED_OPERATION_TEMPLATE_FILE = "selected_operation_template"
    SELECTED_OPERATION_SCAFFOLD_FILE = "selected_operation_scaffold"
    SELECTED_OPERATION_PLAN_FILE = "selected_operation_plan"
    GOAL_REPLAY_PLAN_FILE = "goal_replay_plan"


AuthoringTemplateRegistryKey: TypeAlias = (
    CodemodAuthoringCommandActionId | CodemodAuthoringWorkflowId
)


@dataclass(frozen=True)
class CodemodCliCommandSpec:
    """Directly runnable CLI action emitted by codemod workflow artifacts."""

    action_id: Enum
    args: tuple[str, ...]
    cwd: Path
    module: str = "nominal_refactor_advisor"
    python_executable: str = sys.executable
    required_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...] = ()
    generated_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    generated_artifacts: tuple[str, ...] = ()

    @property
    def argv(self) -> tuple[str, ...]:
        return (
            self.python_executable,
            "-m",
            self.module,
            *self.args,
        )

    def to_dict(self) -> JsonObject:
        return {
            "action_id": str(self.action_id.value),
            "cwd": self.cwd.as_posix(),
            "python_executable": self.python_executable,
            "module": self.module,
            "args": self.args,
            "argv": self.argv,
            "required_artifact_roles": tuple(
                role.value for role in self.required_artifact_roles
            ),
            "generated_artifact_roles": tuple(
                role.value for role in self.generated_artifact_roles
            ),
            "required_artifacts": self.required_artifacts,
            "generated_artifacts": self.generated_artifacts,
        }


@dataclass(frozen=True, kw_only=True)
class CodemodAuthoringWorkflowActionCommandSpec(CodemodCliCommandSpec):
    """Runnable bundle-control command for one workflow target action."""

    workflow_id: CodemodAuthoringWorkflowId
    target_action_id: CodemodAuthoringCommandActionId

    def to_dict(self) -> JsonObject:
        payload = super().to_dict()
        payload.update(
            {
                "workflow_id": self.workflow_id.value,
                "target_action_id": self.target_action_id.value,
            }
        )
        return payload


@dataclass(frozen=True)
class CodemodAuthoringBundleCommandContext:
    """Paths and roots required to emit one replayable authoring command."""

    roots: tuple[Path, ...]
    evidence_selector_file: Path
    replacement_scaffold_file: Path
    replacement_plan_file: Path
    operation_template_file: Path
    selected_operation_scaffold_file: Path
    selected_operation_plan_file: Path
    goal_replay_plan_file: Path
    output_dir: Path
    cwd: Path
    finding_id: str

    @property
    def root_args(self) -> tuple[str, ...]:
        return tuple(root.as_posix() for root in self.roots)

    @property
    def selector_arg(self) -> str:
        return self.evidence_selector_file.as_posix()

    @property
    def plan_arg(self) -> str:
        return self.replacement_plan_file.as_posix()

    @property
    def operation_template_arg(self) -> str:
        return self.operation_template_file.as_posix()

    @property
    def selected_operation_plan_arg(self) -> str:
        return self.selected_operation_plan_file.as_posix()

    @property
    def goal_replay_plan_arg(self) -> str:
        return self.goal_replay_plan_file.as_posix()

    def bundle_relative_path(self, path: Path) -> str:
        try:
            relative_path = path.relative_to(self.output_dir)
        except ValueError as error:
            raise ValueError(
                f"Authoring artifact {path} is outside bundle root {self.output_dir}"
            ) from error
        return relative_path.as_posix()

    def artifact_path(self, role: CodemodAuthoringArtifactRole) -> Path:
        return {
            CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE: (
                self.evidence_selector_file
            ),
            CodemodAuthoringArtifactRole.REPLACEMENT_SCAFFOLD_FILE: (
                self.replacement_scaffold_file
            ),
            CodemodAuthoringArtifactRole.REPLACEMENT_PLAN_FILE: (
                self.replacement_plan_file
            ),
            CodemodAuthoringArtifactRole.SELECTED_OPERATION_TEMPLATE_FILE: (
                self.operation_template_file
            ),
            CodemodAuthoringArtifactRole.SELECTED_OPERATION_SCAFFOLD_FILE: (
                self.selected_operation_scaffold_file
            ),
            CodemodAuthoringArtifactRole.SELECTED_OPERATION_PLAN_FILE: (
                self.selected_operation_plan_file
            ),
            CodemodAuthoringArtifactRole.GOAL_REPLAY_PLAN_FILE: (
                self.goal_replay_plan_file
            ),
        }[role]

    def bundle_relative_paths(
        self,
        roles: tuple[CodemodAuthoringArtifactRole, ...],
    ) -> tuple[str, ...]:
        return tuple(
            self.bundle_relative_path(self.artifact_path(role)) for role in roles
        )


@dataclass(frozen=True)
class CodemodAuthoringWorkflowDefinition:
    """Shared identity and roles for authoring workflow views."""

    workflow_id: CodemodAuthoringWorkflowId
    description: str
    editable_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...]
    review_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...]
    generated_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...]
    command_action_ids: tuple[CodemodAuthoringCommandActionId, ...]
    default_next_action_id: CodemodAuthoringCommandActionId

    def definition_items(self) -> tuple[tuple[str, JsonValue], ...]:
        return (
            ("workflow_id", self.workflow_id.value),
            ("description", self.description),
            (
                "editable_artifact_roles",
                tuple(role.value for role in self.editable_artifact_roles),
            ),
            (
                "review_artifact_roles",
                tuple(role.value for role in self.review_artifact_roles),
            ),
            (
                "generated_artifact_roles",
                tuple(role.value for role in self.generated_artifact_roles),
            ),
            (
                "command_action_ids",
                tuple(action_id.value for action_id in self.command_action_ids),
            ),
            ("default_next_action_id", self.default_next_action_id.value),
        )


@dataclass(frozen=True)
class CodemodAuthoringWorkflowSpec(CodemodAuthoringWorkflowDefinition):
    """Ordered commands and artifacts for one bundle authoring workflow."""

    editable_artifacts: tuple[str, ...]
    review_artifacts: tuple[str, ...]
    generated_artifacts: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        payload = JsonObject(dict(self.definition_items()))
        payload.update(
            {
                "editable_artifacts": self.editable_artifacts,
                "review_artifacts": self.review_artifacts,
                "generated_artifacts": self.generated_artifacts,
            }
        )
        return payload


@dataclass(frozen=True)
class CodemodAuthoringCommandManifest:
    """Manifest row for one registered replayable authoring command."""

    action_id: CodemodAuthoringCommandActionId
    class_name: str
    description: str
    registry_order: int
    required_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...] = ()
    generated_artifact_roles: tuple[CodemodAuthoringArtifactRole, ...] = ()

    def to_dict(self) -> JsonObject:
        return {
            "action_id": self.action_id.value,
            "class_name": self.class_name,
            "description": self.description,
            "registry_order": self.registry_order,
            "required_artifact_roles": tuple(
                role.value for role in self.required_artifact_roles
            ),
            "generated_artifact_roles": tuple(
                role.value for role in self.generated_artifact_roles
            ),
        }


@dataclass(frozen=True)
class CodemodAuthoringWorkflowManifest(CodemodAuthoringWorkflowDefinition):
    """Manifest row for one registered authoring workflow."""

    class_name: str
    registry_order: int

    def to_dict(self) -> JsonObject:
        payload = JsonObject(dict(self.definition_items()))
        payload.update(
            {
                "class_name": self.class_name,
                "registry_order": self.registry_order,
            }
        )
        return payload


@dataclass(frozen=True)
class OrderedAuthoringTemplateRegistry:
    """Ordered view over one authoring template registry."""

    registry: Mapping[AuthoringTemplateRegistryKey, type]

    def template_types(self) -> tuple[type, ...]:
        return tuple(
            sorted(
                self.registry.values(),
                key=lambda template_type: template_type.registry_order,
            )
        )


class CodemodAuthoringBundleWorkflowTemplate(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered workflow emitted into an authoring bundle record."""

    __registry__: ClassVar[
        dict[
            CodemodAuthoringWorkflowId,
            type["CodemodAuthoringBundleWorkflowTemplate"],
        ]
    ] = {}
    __registry_key__ = "workflow_id"
    __skip_if_no_key__ = True

    workflow_id: ClassVar[CodemodAuthoringWorkflowId | None] = None
    registry_order: ClassVar[int]
    description: ClassVar[str]
    command_action_ids: ClassVar[tuple[CodemodAuthoringCommandActionId, ...]]
    default_next_action_id: ClassVar[CodemodAuthoringCommandActionId]
    editable_artifact_roles: ClassVar[tuple[CodemodAuthoringArtifactRole, ...]] = ()
    review_artifact_roles: ClassVar[tuple[CodemodAuthoringArtifactRole, ...]] = ()
    generated_artifact_roles: ClassVar[tuple[CodemodAuthoringArtifactRole, ...]] = ()

    @property
    def required_workflow_id(self) -> CodemodAuthoringWorkflowId:
        if self.workflow_id is None:
            raise RuntimeError(
                "registered authoring workflow template has no workflow_id"
            )
        return self.workflow_id

    def workflow_spec(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> CodemodAuthoringWorkflowSpec:
        return CodemodAuthoringWorkflowSpec(
            workflow_id=self.required_workflow_id,
            description=self.description,
            editable_artifacts=context.bundle_relative_paths(
                self.editable_artifact_roles
            ),
            review_artifacts=context.bundle_relative_paths(self.review_artifact_roles),
            generated_artifacts=context.bundle_relative_paths(
                self.generated_artifact_roles
            ),
            editable_artifact_roles=self.editable_artifact_roles,
            review_artifact_roles=self.review_artifact_roles,
            generated_artifact_roles=self.generated_artifact_roles,
            command_action_ids=self.command_action_ids,
            default_next_action_id=self.default_next_action_id,
        )

    def workflow_manifest(self) -> CodemodAuthoringWorkflowManifest:
        return CodemodAuthoringWorkflowManifest(
            workflow_id=self.required_workflow_id,
            class_name=type(self).__name__,
            description=self.description,
            registry_order=self.registry_order,
            editable_artifact_roles=self.editable_artifact_roles,
            review_artifact_roles=self.review_artifact_roles,
            generated_artifact_roles=self.generated_artifact_roles,
            command_action_ids=self.command_action_ids,
            default_next_action_id=self.default_next_action_id,
        )


class ReplacementPlanAuthoringWorkflowTemplate(CodemodAuthoringBundleWorkflowTemplate):
    workflow_id = CodemodAuthoringWorkflowId.REPLACEMENT_PLAN
    registry_order = 10
    description = (
        "Edit the target replacement plan, then validate, simulate, and apply it."
    )
    command_action_ids = (
        CodemodAuthoringCommandActionId.RESOLVE_SELECTOR,
        CodemodAuthoringCommandActionId.SCAFFOLD_REPLACEMENT_PLAN,
        CodemodAuthoringCommandActionId.VALIDATE_REPLACEMENT_PLAN,
        CodemodAuthoringCommandActionId.SIMULATE_REPLACEMENT_PLAN,
        CodemodAuthoringCommandActionId.APPLY_REPLACEMENT_PLAN,
    )
    default_next_action_id = CodemodAuthoringCommandActionId.SIMULATE_REPLACEMENT_PLAN
    editable_artifact_roles = (CodemodAuthoringArtifactRole.REPLACEMENT_PLAN_FILE,)
    review_artifact_roles = (
        CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,
        CodemodAuthoringArtifactRole.REPLACEMENT_SCAFFOLD_FILE,
    )


class SelectedOperationAuthoringWorkflowTemplate(
    CodemodAuthoringBundleWorkflowTemplate
):
    workflow_id = CodemodAuthoringWorkflowId.SELECTED_OPERATION_TEMPLATE
    registry_order = 20
    description = (
        "Edit the selected-operation template, scaffold a plan, then preflight, "
        "simulate, and apply it."
    )
    command_action_ids = (
        CodemodAuthoringCommandActionId.SCAFFOLD_SELECTED_OPERATION_PLAN,
        CodemodAuthoringCommandActionId.PREFLIGHT_SELECTED_OPERATION_PLAN,
        CodemodAuthoringCommandActionId.SIMULATE_SELECTED_OPERATION_PLAN,
        CodemodAuthoringCommandActionId.APPLY_SELECTED_OPERATION_PLAN,
    )
    default_next_action_id = (
        CodemodAuthoringCommandActionId.SIMULATE_SELECTED_OPERATION_PLAN
    )
    editable_artifact_roles = (
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_TEMPLATE_FILE,
    )
    review_artifact_roles = (
        CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_SCAFFOLD_FILE,
    )
    generated_artifact_roles = (
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_PLAN_FILE,
    )


class GoalRefactorAuthoringWorkflowTemplate(CodemodAuthoringBundleWorkflowTemplate):
    workflow_id = CodemodAuthoringWorkflowId.GOAL_REFACTOR
    registry_order = 30
    description = (
        "Run a goal-directed finding refactor, then simulate or apply the generated "
        "staged replay plan."
    )
    command_action_ids = (
        CodemodAuthoringCommandActionId.RUN_GOAL_REFACTOR,
        CodemodAuthoringCommandActionId.SIMULATE_GOAL_REPLAY_PLAN,
        CodemodAuthoringCommandActionId.APPLY_GOAL_REPLAY_PLAN,
    )
    default_next_action_id = CodemodAuthoringCommandActionId.RUN_GOAL_REFACTOR
    review_artifact_roles = (CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,)
    generated_artifact_roles = (CodemodAuthoringArtifactRole.GOAL_REPLAY_PLAN_FILE,)


class CodemodAuthoringBundleCommandTemplate(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered command template emitted into an authoring bundle."""

    __registry__: ClassVar[
        dict[
            CodemodAuthoringCommandActionId,
            type["CodemodAuthoringBundleCommandTemplate"],
        ]
    ] = {}
    __registry_key__ = "action_id"
    __skip_if_no_key__ = True

    action_id: ClassVar[CodemodAuthoringCommandActionId | None] = None
    registry_order: ClassVar[int]
    description: ClassVar[str] = ""
    required_artifact_roles: ClassVar[tuple[CodemodAuthoringArtifactRole, ...]] = ()
    generated_artifact_roles: ClassVar[tuple[CodemodAuthoringArtifactRole, ...]] = ()

    def command_spec(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> CodemodCliCommandSpec:
        if self.action_id is None:
            raise RuntimeError("registered authoring command template has no action_id")
        return CodemodCliCommandSpec(
            action_id=self.action_id,
            args=self.command_args(context),
            cwd=context.cwd,
            required_artifact_roles=self.required_artifact_roles,
            generated_artifact_roles=self.generated_artifact_roles,
            required_artifacts=context.bundle_relative_paths(
                self.required_artifact_roles
            ),
            generated_artifacts=context.bundle_relative_paths(
                self.generated_artifact_roles
            ),
        )

    def command_manifest(self) -> CodemodAuthoringCommandManifest:
        if self.action_id is None:
            raise RuntimeError("registered authoring command template has no action_id")
        return CodemodAuthoringCommandManifest(
            action_id=self.action_id,
            class_name=type(self).__name__,
            description=self.description,
            registry_order=self.registry_order,
            required_artifact_roles=self.required_artifact_roles,
            generated_artifact_roles=self.generated_artifact_roles,
        )

    @abstractmethod
    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        raise NotImplementedError


class ResolveSelectorCommandTemplate(CodemodAuthoringBundleCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.RESOLVE_SELECTOR
    registry_order = 10
    required_artifact_roles = (CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-resolve-selector",
            context.selector_arg,
        )


class ScaffoldReplacementPlanCommandTemplate(CodemodAuthoringBundleCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.SCAFFOLD_REPLACEMENT_PLAN
    registry_order = 20
    required_artifact_roles = (CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,)
    generated_artifact_roles = (CodemodAuthoringArtifactRole.REPLACEMENT_PLAN_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-replacement-plan",
            context.selector_arg,
            "--codemod-plan-out",
            context.plan_arg,
        )


class ValidateReplacementPlanCommandTemplate(CodemodAuthoringBundleCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.VALIDATE_REPLACEMENT_PLAN
    registry_order = 30
    required_artifact_roles = (CodemodAuthoringArtifactRole.REPLACEMENT_PLAN_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            "--codemod-plan",
            context.plan_arg,
            "--codemod-validate-plan",
        )


class ReplacementPlanExecutionCommandTemplate(
    CodemodAuthoringBundleCommandTemplate,
    ABC,
):
    execution_flag: ClassVar[str]
    required_artifact_roles = (CodemodAuthoringArtifactRole.REPLACEMENT_PLAN_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-plan",
            context.plan_arg,
            self.execution_flag,
        )


class SimulateReplacementPlanCommandTemplate(ReplacementPlanExecutionCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.SIMULATE_REPLACEMENT_PLAN
    registry_order = 40
    execution_flag = "--codemod-simulate"


class ApplyReplacementPlanCommandTemplate(ReplacementPlanExecutionCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.APPLY_REPLACEMENT_PLAN
    registry_order = 50
    execution_flag = "--codemod-apply"


class ScaffoldSelectedOperationPlanCommandTemplate(
    CodemodAuthoringBundleCommandTemplate
):
    action_id = CodemodAuthoringCommandActionId.SCAFFOLD_SELECTED_OPERATION_PLAN
    registry_order = 60
    required_artifact_roles = (
        CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_TEMPLATE_FILE,
    )
    generated_artifact_roles = (
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_PLAN_FILE,
    )

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-selected-operation-plan",
            context.selector_arg,
            "--codemod-operation-template",
            context.operation_template_arg,
            "--codemod-plan-out",
            context.selected_operation_plan_arg,
        )


class SelectedOperationPlanExecutionCommandTemplate(
    CodemodAuthoringBundleCommandTemplate,
    ABC,
):
    execution_flag: ClassVar[str]
    required_artifact_roles = (
        CodemodAuthoringArtifactRole.EVIDENCE_SELECTOR_FILE,
        CodemodAuthoringArtifactRole.SELECTED_OPERATION_TEMPLATE_FILE,
    )

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-selected-operation-plan",
            context.selector_arg,
            "--codemod-operation-template",
            context.operation_template_arg,
            self.execution_flag,
        )


class PreflightSelectedOperationPlanCommandTemplate(
    SelectedOperationPlanExecutionCommandTemplate
):
    action_id = CodemodAuthoringCommandActionId.PREFLIGHT_SELECTED_OPERATION_PLAN
    registry_order = 70
    execution_flag = "--codemod-preflight"


class SimulateSelectedOperationPlanCommandTemplate(
    SelectedOperationPlanExecutionCommandTemplate
):
    action_id = CodemodAuthoringCommandActionId.SIMULATE_SELECTED_OPERATION_PLAN
    registry_order = 80
    execution_flag = "--codemod-simulate"


class ApplySelectedOperationPlanCommandTemplate(
    SelectedOperationPlanExecutionCommandTemplate
):
    action_id = CodemodAuthoringCommandActionId.APPLY_SELECTED_OPERATION_PLAN
    registry_order = 90
    execution_flag = "--codemod-apply"


class RunGoalRefactorCommandTemplate(CodemodAuthoringBundleCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.RUN_GOAL_REFACTOR
    registry_order = 100
    generated_artifact_roles = (CodemodAuthoringArtifactRole.GOAL_REPLAY_PLAN_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-refactor-goal",
            CodemodRefactorGoalKind.NOMINAL_BOUNDARY_EXTRACTION.value,
            "--codemod-goal-finding-id",
            context.finding_id,
            "--codemod-goal-plan-out",
            context.goal_replay_plan_arg,
            "--json",
        )


class GoalReplayPlanExecutionCommandTemplate(
    CodemodAuthoringBundleCommandTemplate,
    ABC,
):
    execution_flag: ClassVar[str]
    required_artifact_roles = (CodemodAuthoringArtifactRole.GOAL_REPLAY_PLAN_FILE,)

    def command_args(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return (
            *context.root_args,
            "--codemod-plan",
            context.goal_replay_plan_arg,
            self.execution_flag,
        )


class SimulateGoalReplayPlanCommandTemplate(GoalReplayPlanExecutionCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.SIMULATE_GOAL_REPLAY_PLAN
    registry_order = 110
    execution_flag = "--codemod-simulate"


class ApplyGoalReplayPlanCommandTemplate(GoalReplayPlanExecutionCommandTemplate):
    action_id = CodemodAuthoringCommandActionId.APPLY_GOAL_REPLAY_PLAN
    registry_order = 120
    execution_flag = "--codemod-apply"


@dataclass(frozen=True)
class CodemodAuthoringBundleWriter:
    """Materialize per-finding synthesis authoring artifacts for agents."""

    output_dir: Path
    snapshot: CodemodSourceSnapshot
    plan: FindingRecipePlan
    roots: tuple[Path, ...]
    cwd: Path

    @property
    def index_path(self) -> Path:
        return self.output_dir / "index.json"

    def write(self) -> JsonObject:
        records = tuple(
            self.write_record(record_index, record)
            for record_index, record in enumerate(self.plan.records)
        )
        payload: JsonObject = {
            **self.plan.synthesis_payload(),
            "bundle_commands": self.bundle_control_commands(),
            "records": records,
        }
        write_cli_json_artifact(self.index_path, payload)
        return payload

    def write_record(
        self,
        record_index: int,
        record: FindingRecipeSynthesisRecord,
    ) -> JsonObject:
        authoring_record = record.authoring_record()
        record_dir = self.record_dir(record_index, authoring_record.detector_id)
        selector_payload = authoring_record.evidence_selector.to_dict()
        scaffold = self.snapshot.replacement_plan_scaffold_report(
            authoring_record.evidence_selector
        )
        selector_path = record_dir / "selector.json"
        scaffold_path = record_dir / "replacement-scaffold.json"
        plan_path = record_dir / "replacement-plan.json"
        operation_template_path = record_dir / "selected-operation-template.json"
        selected_scaffold_path = record_dir / "selected-operation-scaffold.json"
        selected_plan_path = record_dir / "selected-operation-plan.json"
        goal_replay_plan_path = record_dir / "goal-replay-plan.json"
        operation_template = self.selected_operation_plan_template()
        selected_scaffold = self.snapshot.selected_operation_plan_scaffold_report(
            authoring_record.evidence_selector,
            operation_template,
        )
        write_cli_json_artifact(selector_path, selector_payload)
        write_cli_json_artifact(scaffold_path, scaffold.to_dict())
        write_cli_json_artifact(plan_path, scaffold.document.to_dict())
        write_cli_json_artifact(operation_template_path, operation_template.to_dict())
        write_cli_json_artifact(selected_scaffold_path, selected_scaffold.to_dict())
        write_cli_json_artifact(
            selected_plan_path, selected_scaffold.document.to_dict()
        )
        context = self.authoring_context(
            authoring_record.finding_id,
            selector_path,
            scaffold_path,
            plan_path,
            operation_template_path,
            selected_scaffold_path,
            selected_plan_path,
            goal_replay_plan_path,
        )
        command_specs = self.command_specs(context)
        workflow_specs = self.workflow_specs(context)
        command_payloads = tuple(command.to_dict() for command in command_specs)
        workflow_payloads = tuple(workflow.to_dict() for workflow in workflow_specs)
        workflow_readiness = (
            CodemodAuthoringWorkflowPlanner.from_payloads(
                command_payloads,
                workflow_payloads,
            )
            .bundle_readiness(
                self.available_authoring_artifacts(context),
            )
            .to_dict()
        )
        return {
            "record_index": record_index,
            "finding_id": authoring_record.finding_id,
            "detector_id": authoring_record.detector_id,
            "status": authoring_record.status.value,
            "selector_path": selector_path.relative_to(self.output_dir).as_posix(),
            "replacement_scaffold_path": scaffold_path.relative_to(
                self.output_dir
            ).as_posix(),
            "replacement_plan_path": plan_path.relative_to(self.output_dir).as_posix(),
            "selected_operation_template_path": operation_template_path.relative_to(
                self.output_dir
            ).as_posix(),
            "selected_operation_scaffold_path": selected_scaffold_path.relative_to(
                self.output_dir
            ).as_posix(),
            "selected_operation_plan_path": selected_plan_path.relative_to(
                self.output_dir
            ).as_posix(),
            "goal_replay_plan_path": goal_replay_plan_path.relative_to(
                self.output_dir
            ).as_posix(),
            "commands": command_payloads,
            "workflows": workflow_payloads,
            "workflow_action_commands": tuple(
                command.to_dict()
                for command in self.workflow_action_command_specs(
                    record_index,
                    workflow_specs,
                )
            ),
            "workflow_readiness": workflow_readiness,
            "authoring_record": authoring_record.to_dict(),
        }

    def record_dir(self, record_index: int, detector_id: str) -> Path:
        return self.output_dir / f"{record_index:04d}-{cli_artifact_slug(detector_id)}"

    def authoring_context(
        self,
        finding_id: str,
        evidence_selector_file: Path,
        replacement_scaffold_file: Path,
        replacement_plan_file: Path,
        operation_template_file: Path,
        selected_operation_scaffold_file: Path,
        selected_operation_plan_file: Path,
        goal_replay_plan_file: Path,
    ) -> CodemodAuthoringBundleCommandContext:
        return CodemodAuthoringBundleCommandContext(
            roots=self.roots,
            evidence_selector_file=evidence_selector_file,
            replacement_scaffold_file=replacement_scaffold_file,
            replacement_plan_file=replacement_plan_file,
            operation_template_file=operation_template_file,
            selected_operation_scaffold_file=selected_operation_scaffold_file,
            selected_operation_plan_file=selected_operation_plan_file,
            goal_replay_plan_file=goal_replay_plan_file,
            output_dir=self.output_dir,
            cwd=self.cwd,
            finding_id=finding_id,
        )

    def command_specs(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[CodemodCliCommandSpec, ...]:
        return tuple(
            template_type().command_spec(context)
            for template_type in OrderedAuthoringTemplateRegistry(
                CodemodAuthoringBundleCommandTemplate.__registry__
            ).template_types()
        )

    def bundle_control_commands(self) -> JsonObject:
        return {
            "status": self.status_command_spec().to_dict(),
        }

    def status_command_spec(self) -> CodemodCliCommandSpec:
        return CodemodCliCommandSpec(
            action_id=CodemodAuthoringBundleControlActionId.AUTHORING_STATUS,
            args=(
                "--codemod-authoring-status",
                self.index_path.as_posix(),
            ),
            cwd=self.cwd,
        )

    def workflow_action_command_specs(
        self,
        record_index: int,
        workflow_specs: tuple[CodemodAuthoringWorkflowSpec, ...],
    ) -> tuple[CodemodAuthoringWorkflowActionCommandSpec, ...]:
        return tuple(
            CodemodAuthoringWorkflowActionCommandSpec(
                workflow_id=workflow.workflow_id,
                target_action_id=action_id,
                action_id=CodemodAuthoringBundleControlActionId.AUTHORING_RUN_ACTION,
                args=self.workflow_action_command_args(
                    record_index,
                    workflow.workflow_id,
                    action_id,
                ),
                cwd=self.cwd,
            )
            for workflow in workflow_specs
            for action_id in workflow.command_action_ids
        )

    def workflow_action_command_args(
        self,
        record_index: int,
        workflow_id: CodemodAuthoringWorkflowId,
        target_action_id: CodemodAuthoringCommandActionId,
    ) -> tuple[str, ...]:
        return (
            "--codemod-authoring-run-action",
            self.index_path.as_posix(),
            "--codemod-authoring-record-index",
            str(record_index),
            "--codemod-authoring-workflow-id",
            workflow_id.value,
            "--codemod-authoring-target-action",
            target_action_id.value,
        )

    def workflow_specs(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[CodemodAuthoringWorkflowSpec, ...]:
        return tuple(
            template_type().workflow_spec(context)
            for template_type in OrderedAuthoringTemplateRegistry(
                CodemodAuthoringBundleWorkflowTemplate.__registry__
            ).template_types()
        )

    def available_authoring_artifacts(
        self,
        context: CodemodAuthoringBundleCommandContext,
    ) -> tuple[str, ...]:
        return tuple(
            context.bundle_relative_path(path)
            for path in (
                context.artifact_path(role) for role in CodemodAuthoringArtifactRole
            )
            if path.exists()
        )

    @staticmethod
    def selected_operation_plan_template() -> RefactorRecipeOperationPlanTemplate:
        return RefactorRecipeOperationPlanTemplate.from_operation_templates(
            (
                RefactorRecipeOperationTemplate.from_payload(
                    {
                        "operation": RefactorRecipeOperationKind.REPLACE_TEXT.value,
                        OLD_SOURCE_PAYLOAD_FIELD: "${target.source}",
                        NEW_SOURCE_PAYLOAD_FIELD: "${target.source}",
                    }
                ),
            )
        )


def codemod_authoring_command_manifest_payloads() -> tuple[JsonObject, ...]:
    """Return manifest rows for registered authoring bundle command actions."""

    return tuple(
        template_type().command_manifest().to_dict()
        for template_type in OrderedAuthoringTemplateRegistry(
            CodemodAuthoringBundleCommandTemplate.__registry__
        ).template_types()
    )


def codemod_authoring_workflow_manifest_payloads() -> tuple[JsonObject, ...]:
    """Return manifest rows for registered authoring bundle workflows."""

    return tuple(
        template_type().workflow_manifest().to_dict()
        for template_type in OrderedAuthoringTemplateRegistry(
            CodemodAuthoringBundleWorkflowTemplate.__registry__
        ).template_types()
    )


def codemod_selected_operation_template_source_payloads() -> tuple[JsonObject, ...]:
    """Return manifest rows for selected-operation template sources."""

    return tuple(
        source_type().source_manifest().to_dict()
        for source_type in SelectedOperationTemplateSource.ordered_source_types()
    )


def codemod_selected_operation_target_selector_source_payloads() -> (
    tuple[JsonObject, ...]
):
    """Return manifest rows for selected-operation target selector sources."""

    return tuple(
        source_type().source_manifest().to_dict()
        for source_type in SelectedOperationTargetSelectorSource.ordered_source_types()
    )


def codemod_cli_dsl_manifest_payload() -> JsonObject:
    """Return the codemod DSL manifest plus executable authoring workflow metadata."""

    payload = codemod_dsl_manifest().to_dict()
    payload["workflow_plans"] = tuple(
        manifest.to_dict() for manifest in codemod_workflow_plan_manifests()
    )
    payload["workflow_plan_examples"] = codemod_workflow_plan_example_payloads()
    payload["authoring_artifact_roles"] = tuple(
        artifact_role.value for artifact_role in CodemodAuthoringArtifactRole
    )
    payload["authoring_command_actions"] = codemod_authoring_command_manifest_payloads()
    payload["authoring_workflows"] = codemod_authoring_workflow_manifest_payloads()
    payload["selected_operation_target_selector_sources"] = (
        codemod_selected_operation_target_selector_source_payloads()
    )
    payload["selected_operation_template_sources"] = (
        codemod_selected_operation_template_source_payloads()
    )
    return payload


@dataclass(frozen=True)
class CodemodSimulationPayload:
    """JSON-ready metadata for a codemod simulation/apply run."""

    simulation: CodemodSimulationReport
    applied: bool = False
    post_guard_report: ArchitectureGuardReport | None = None
    unified_diff: str | None = None

    def to_dict(self) -> JsonObject:
        payload = self.simulation.to_dict()
        payload["applied"] = self.applied
        if self.post_guard_report is not None:
            payload["architecture_guard_report"] = self.post_guard_report.to_dict()
        if self.unified_diff is not None:
            payload["unified_diff"] = self.unified_diff
        return payload


@dataclass(frozen=True)
class CodemodPreflightFailurePayload:
    """JSON-ready metadata for a codemod preflight failure."""

    report: CodemodOperationPreflightReport

    def to_dict(self) -> JsonObject:
        return {
            "preflight_failed": True,
            "applied": False,
            "preflight_report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class CodemodPlanPreflightPayload:
    """JSON-ready metadata for codemod plan preflight mode."""

    report: CodemodPlanPreflightReport

    def to_dict(self) -> JsonObject:
        return {
            **self.report.to_dict(),
            "applied": False,
        }


class CodemodProjectedFindingReporter(ABC):
    """Mixin for commands that can rescan simulated source states."""

    args: argparse.Namespace
    modules: list[ParsedModule]
    findings: list[RefactorFinding]
    config: DetectorConfig
    roots: tuple[Path, ...]

    def optional_projected_finding_report(
        self,
        simulation: CodemodSimulationReport,
        *,
        enabled: bool,
        source_sequence: CodemodPlanSequence | None = None,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> CodemodProjectedFindingReport | None:
        if not enabled:
            return None
        return CodemodSimulationFindingProjection(
            modules=tuple(self.modules),
            findings=tuple(self.findings),
            simulation=simulation,
            config=self.config,
            roots=self.roots,
            source_sequence=source_sequence,
            expected_removed_finding_ids=expected_removed_finding_ids,
        ).report()

    def write_continuation_plan_if_requested(
        self,
        report: CodemodProjectedFindingReport,
    ) -> None:
        write_cli_json_artifact(
            self.args.codemod_continuation_plan_out,
            report.continuation_report.continuation_sequence.to_dict(),
        )


def format_codemod_fixpoint_markdown(report: CodemodFixpointReport) -> str:
    """Render a concise fixpoint workflow summary."""

    return report.to_markdown()


def format_codemod_refactor_goal_markdown(
    report: CodemodRefactorGoalReport,
) -> str:
    """Render a concise goal-directed codemod workflow summary."""

    return report.to_markdown()


def codemod_refactor_goal_from_args(
    args: argparse.Namespace,
) -> CodemodRefactorGoal:
    """Build the high-level codemod goal requested by CLI flags."""

    try:
        goal_kind = CodemodRefactorGoalKind(args.codemod_refactor_goal)
    except ValueError as error:
        choices = ", ".join(item.value for item in CodemodRefactorGoalKind)
        raise ValueError(
            f"unknown codemod refactor goal {args.codemod_refactor_goal!r}; "
            f"choose one of {choices}"
        ) from error
    return CodemodRefactorGoal(
        goal_id=goal_kind.value,
        kind=goal_kind,
        target_finding_ids=tuple(args.codemod_goal_finding_ids),
        detector_ids=tuple(args.codemod_goal_detectors),
        pattern_ids=tuple(args.codemod_goal_patterns),
        max_stages=args.codemod_goal_max_stages,
    )


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
        pattern_sequence = plan.pattern_sequence
        primary = PATTERN_SPECS[pattern_sequence.primary_pattern_id]
        order = " -> ".join(
            (
                f"Pattern {pattern_id.value}"
                for pattern_id in pattern_sequence.ordered_pattern_ids
            )
        )
        lines.append(f"{index}. {plan.subsystem}")
        lines.append(f"   - Summary: {plan.summary}")
        lines.append(
            f"   - Primary pattern: Pattern {primary.pattern_id.value}: {primary.name}"
        )
        if pattern_sequence.secondary_pattern_ids:
            secondary = ", ".join(
                (
                    f"Pattern {pattern_id.value}: {PATTERN_SPECS[pattern_id].name}"
                    for pattern_id in pattern_sequence.secondary_pattern_ids
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
        pattern_sequence = execution_class.pattern_sequence
        primary = PATTERN_SPECS[pattern_sequence.primary_pattern_id]
        order = " -> ".join(
            (
                f"Pattern {pattern_id.value}"
                for pattern_id in pattern_sequence.ordered_pattern_ids
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
            candidate.applicability.strategy.automation_level
            is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
            for candidate in candidates
        )
    )
    safe_count = sum(
        (candidate.applicability.strategy.safe_to_apply for candidate in candidates)
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
            f"   - Candidate {index}: {applicability.strategy.automation_level.value} "
            f"`{candidate.opportunity_key.label}` -> "
            f"{candidate.target_count} target(s), "
            f"{candidate.predicted_removed_finding_count} finding(s), "
            f"{applicability.planned_rewrite_count} planned rewrite(s), "
            f"simulation {applicability.simulation_status.value}"
        )
        lines.append(f"     strategy: {applicability.strategy.strategy_id}")
        lines.append(f"     actionability: {applicability.actionability.value}")
        lines.append(f"     confidence basis: {applicability.confidence_basis}")
        lines.append(f"     reason: {applicability.strategy.reason}")
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
class CodemodSynthesisExitCodeAuthority:
    """Exit-code policy for one synthesized codemod batch."""

    is_clean: bool

    def exit_code(self) -> int:
        if self.is_clean:
            return 0
        return 1


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
        print(json.dumps(codemod_cli_dsl_manifest_payload(), indent=2))
        return 0


class CodemodDslExamplePlanCliCommand(CliEarlyExitCommand):
    """Emit a registry-derived codemod DSL starter plan."""

    command_id = "codemod_dsl_example_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_dsl_example_plan

    def run(self) -> int:
        payload = codemod_dsl_example_plan_payload()
        write_cli_json_artifact(self.args.codemod_plan_out, payload)
        print(json.dumps(payload, indent=2))
        return 0


class CodemodAuthoringStatusCliCommand(CliEarlyExitCommand):
    """Recompute authoring bundle workflow readiness."""

    command_id = "codemod_authoring_status"

    @property
    def requested(self) -> bool:
        return self.args.codemod_authoring_status is not None

    def run(self) -> int:
        try:
            payload = (
                CodemodAuthoringBundleStatusReporter.from_index_path(
                    self.args.codemod_authoring_status
                )
                .status()
                .to_dict()
            )
        except (OSError, json.JSONDecodeError, TypeError) as error:
            self.parser.error(str(error))
        print(json.dumps(payload, indent=2))
        return 0


class CodemodAuthoringRunActionCliCommand(CliEarlyExitCommand):
    """Run a planned command chain from an authoring bundle."""

    command_id = "codemod_authoring_run_action"

    @property
    def requested(self) -> bool:
        return self.args.codemod_authoring_run_action is not None

    def run(self) -> int:
        if self.args.codemod_authoring_target_action is None:
            self.parser.error(
                "--codemod-authoring-run-action requires "
                "--codemod-authoring-target-action"
            )
        try:
            report = CodemodAuthoringBundleActionRunner(
                bundle_index_path=self.args.codemod_authoring_run_action,
                record_index=self.args.codemod_authoring_record_index,
                workflow_id=self.args.codemod_authoring_workflow_id,
                target_action_id=self.args.codemod_authoring_target_action,
            ).run()
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            IndexError,
            KeyError,
        ) as error:
            self.parser.error(str(error))
        print(json.dumps(report.to_dict(), indent=2))
        return report.exit_code


class CodemodValidatePlanCliCommand(CliEarlyExitCommand):
    """Validate a supplied codemod DSL plan and emit its normalized form."""

    command_id = "codemod_validate_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_validate_plan

    def run(self) -> int:
        payload = load_codemod_plan_validation_payload(self.plan_path)
        write_cli_json_artifact(self.args.codemod_plan_out, payload)
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return 0

    @property
    def plan_path(self) -> Path:
        if self.args.codemod_plan is None:
            self.parser.error("--codemod-validate-plan requires --codemod-plan")
        return self.args.codemod_plan


class CodemodComposePlansCliCommand(CliEarlyExitCommand):
    """Compose normalized codemod DSL plan documents."""

    command_id = "codemod_compose_plans"

    @property
    def requested(self) -> bool:
        return self.args.codemod_compose_plans is not None

    def run(self) -> int:
        if self.args.codemod_compose_sequence is not None:
            self.parser.error(
                "--codemod-compose-plans cannot be combined with "
                "--codemod-compose-sequence"
            )
        paths = tuple(self.args.codemod_compose_plans)
        JsonDocumentInputSet.from_option_paths(
            (("--codemod-compose-plans", paths),)
        ).require_at_most_one_stdin(self.parser)
        try:
            document = CodemodPlanDocument.compose(
                load_codemod_plan_document(path) for path in paths
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        payload = document.to_dict()
        write_cli_json_artifact(self.args.codemod_plan_out, payload)
        print(json.dumps(payload, indent=2))
        return 0


class CodemodComposeSequenceCliCommand(CliEarlyExitCommand):
    """Compose normalized codemod DSL plans as ordered replay stages."""

    command_id = "codemod_compose_sequence"

    @property
    def requested(self) -> bool:
        return self.args.codemod_compose_sequence is not None

    def run(self) -> int:
        paths = tuple(self.args.codemod_compose_sequence)
        JsonDocumentInputSet.from_option_paths(
            (("--codemod-compose-sequence", paths),)
        ).require_at_most_one_stdin(self.parser)
        try:
            sequence = CodemodPlanSequence.compose(
                load_codemod_plan_sequence(path) for path in paths
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        payload = sequence.to_dict()
        write_cli_json_artifact(self.args.codemod_plan_out, payload)
        print(json.dumps(payload, indent=2))
        return 0


@dataclass(frozen=True)
class ParseCacheDirAuthority(ParseCachePolicy):
    """Resolve the effective parse cache directory for one CLI root."""

    root: Path
    requested_parse_cache_dir: Path | None

    def parse_cache_dir(self) -> Path | None:
        if not self.use_parse_cache:
            return None
        if self.requested_parse_cache_dir is not None:
            return self.requested_parse_cache_dir
        return default_parse_cache_dir(self.root)


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
        known_file_paths = set()
        for parsed_module in self.modules:
            file_path = str(parsed_module.path)
            known_file_paths.add(file_path)
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
        for file_path, source in sorted(source_by_path.items()):
            if file_path in known_file_paths:
                continue
            path = Path(file_path)
            updated_modules.append(
                ParsedModule(
                    path,
                    module_name_from_source_path(file_path),
                    path.name == "__init__.py",
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
class CodemodCliExecution(
    CodemodSourceSnapshotRequiredCommand, CodemodProjectedFindingReporter
):
    """Run the CLI codemod simulation/apply phase through plan-level DSL APIs."""

    command_id = "codemod_execution"
    source_snapshot_error_message: ClassVar[str] = (
        "--codemod-diff/--codemod-preflight/--codemod-simulate/"
        "--codemod-apply require codemod candidates or recipe rewrites"
    )
    impact_candidates: CodemodCandidateSelection
    execution_request: "CodemodPlanExecutionRequest"
    architecture_guard_evaluator: ArchitectureGuardSourceEvaluator
    modules: list[ParsedModule]
    findings: list[RefactorFinding]
    config: DetectorConfig
    roots: tuple[Path, ...]

    @property
    def requested(self) -> bool:
        return self.execution_request.mode.requested

    def run(self) -> int | None:
        if not self.requested:
            return None
        snapshot = self.required_source_snapshot()
        if self.execution_request.mode.preflight:
            return self.emit_preflight_report(
                self.execution_request.sequence.preflight_snapshot(snapshot)
            )
        try:
            simulation, architecture_guard_report, plan_sequence_simulation = (
                self.simulation_context(snapshot)
            )
        except CodemodOperationPreflightError as error:
            return self.emit_preflight_failure(error.report)
        if (
            architecture_guard_report is not None
            and not architecture_guard_report.is_clean
        ):
            return self.emit_guard_failure(
                snapshot,
                simulation,
                architecture_guard_report,
            )
        applied = self.apply_if_requested(simulation, plan_sequence_simulation)
        self.emit_success(
            snapshot,
            simulation,
            applied,
            architecture_guard_report,
            plan_sequence_simulation,
        )
        return 0

    def simulation_context(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[
        CodemodSimulationReport,
        ArchitectureGuardReport | None,
        CodemodPlanSequenceSimulation | None,
    ]:
        if not self.impact_candidates and self.execution_request.sequence.has_recipes:
            plan_sequence_simulation = self.execution_request.sequence.simulate_snapshot(
                snapshot
            )
            return (
                plan_sequence_simulation.simulation,
                self.plan_sequence_guard_report(plan_sequence_simulation),
                plan_sequence_simulation,
            )
        candidate_simulation = snapshot.simulate_rewrites(
            self.candidate_rewrite_batch()
        )
        active_snapshot = snapshot.with_simulation(candidate_simulation)
        if self.execution_request.sequence.has_recipes:
            plan_sequence_simulation = self.execution_request.sequence.simulate_snapshot(
                active_snapshot
            )
            simulation = CodemodSimulationReport.combine(
                (candidate_simulation, plan_sequence_simulation.simulation)
            )
        else:
            plan_sequence_simulation = None
            simulation = candidate_simulation
        return (
            simulation,
            self.architecture_guard_evaluator.report_for_snapshot(
                snapshot.with_simulation(simulation),
            ),
            None,
        )

    def emit_preflight_failure(
        self,
        report: CodemodOperationPreflightReport,
    ) -> int:
        if self.execution_request.mode.json_report_requested(self.args.json):
            print(
                json.dumps(
                    CodemodPreflightFailurePayload(report).to_dict(),
                    indent=2,
                )
            )
        else:
            print(f"Codemod preflight failed: {report.message}", file=sys.stderr)
        return 1

    def emit_preflight_report(
        self,
        report: CodemodPlanPreflightReport,
    ) -> int:
        print(
            json.dumps(
                CodemodPlanPreflightPayload(report).to_dict(),
                indent=2,
            )
        )
        if report.is_clean:
            return 0
        return 1

    def candidate_rewrite_batch(self) -> tuple[PlannedSourceRewrite, ...]:
        if self.impact_candidates is None:
            return ()
        return tuple(
            rewrite
            for candidate in self.impact_candidates
            for rewrite in candidate.planned_rewrites
        )

    def plan_sequence_guard_report(
        self,
        plan_sequence_simulation: CodemodPlanSequenceSimulation,
    ) -> ArchitectureGuardReport | None:
        if not self.execution_request.sequence.has_architecture_guards:
            return None
        return plan_sequence_simulation.architecture_guard_report

    def emit_guard_failure(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        architecture_guard_report: ArchitectureGuardReport,
    ) -> int:
        if self.execution_request.mode.json_report_requested(self.args.json):
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
            if self.execution_request.mode.diff_text_requested:
                print(
                    snapshot.unified_diff(simulation),
                    end="",
                )
            print(format_architecture_guard_markdown(architecture_guard_report))
        return 1

    def apply_if_requested(
        self,
        simulation: CodemodSimulationReport,
        plan_sequence_simulation: CodemodPlanSequenceSimulation | None,
    ) -> bool:
        if not self.execution_request.mode.apply:
            return False
        if plan_sequence_simulation is not None:
            plan_sequence_simulation.apply()
        else:
            apply_codemod_simulation(simulation)
        return True

    def emit_success(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        applied: bool,
        architecture_guard_report: ArchitectureGuardReport | None,
        plan_sequence_simulation: CodemodPlanSequenceSimulation | None,
    ) -> None:
        if self.execution_request.mode.json_report_requested(self.args.json):
            payload = CodemodSimulationPayload(
                simulation,
                applied=applied,
                post_guard_report=architecture_guard_report,
                unified_diff=self.optional_unified_diff(snapshot, simulation),
            ).to_dict()
            if plan_sequence_simulation is not None:
                payload["plan_sequence_simulation"] = plan_sequence_simulation.to_dict()
            projected_findings = self.optional_projected_finding_report(
                simulation,
                enabled=self.execution_request.mode.project_findings,
                source_sequence=(
                    plan_sequence_simulation.sequence
                    if plan_sequence_simulation is not None
                    else None
                ),
            )
            if projected_findings is not None:
                payload["projected_findings"] = projected_findings.to_dict()
                self.write_continuation_plan_if_requested(projected_findings)
            print(
                json.dumps(
                    payload,
                    indent=2,
                )
            )
        elif self.execution_request.mode.diff_text_requested:
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
        if not self.execution_request.mode.unified_diff_requested:
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
    preflight: bool
    simulate: bool
    apply: bool
    fixpoint: bool
    project_findings: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CodemodExecutionMode":
        return cls(
            diff=args.codemod_diff,
            preflight=args.codemod_preflight,
            simulate=args.codemod_simulate,
            apply=args.codemod_apply,
            fixpoint=args.codemod_fixpoint,
            project_findings=args.codemod_project_findings,
        )

    @property
    def requested(self) -> bool:
        return self.mode_count > 0

    @property
    def mode_count(self) -> int:
        return sum((self.diff, self.preflight, self.simulate, self.apply))

    @property
    def unified_diff_requested(self) -> bool:
        return self.diff or self.simulate

    @property
    def diff_text_requested(self) -> bool:
        return self.diff

    def json_report_requested(self, json_flag: bool) -> bool:
        return json_flag or self.preflight or self.simulate or self.project_findings

    def require_valid(self, parser: argparse.ArgumentParser) -> None:
        if self.project_findings and not self.simulate:
            parser.error("--codemod-project-findings requires --codemod-simulate")
        if self.fixpoint and self.diff:
            parser.error("--codemod-fixpoint cannot be combined with --codemod-diff")
        if self.fixpoint and self.preflight:
            parser.error(
                "--codemod-fixpoint cannot be combined with --codemod-preflight"
            )
        if self.fixpoint and self.simulate:
            parser.error(
                "--codemod-fixpoint cannot be combined with --codemod-simulate"
            )
        if self.mode_count <= 1:
            return
        parser.error(
            "--codemod-diff, --codemod-preflight, --codemod-simulate, and "
            "--codemod-apply are mutually exclusive"
        )


@dataclass(frozen=True)
class CodemodPlanExecutionRequest:
    """Codemod plan plus execution mode consumed by execution authorities."""

    sequence: CodemodPlanSequence
    mode: CodemodExecutionMode

    @property
    def exact_recipe_execution(self) -> bool:
        return (
            self.mode.requested
            and not self.mode.project_findings
            and self.sequence.has_recipes
            and not self.sequence.has_authority_boundaries
            and not self.sequence.has_architecture_guards
        )


@dataclass(frozen=True)
class SelectedOperationTemplateSourceSelection:
    """Validated pairing between selected-operation selector and template source."""

    selector_source_label: str
    selector_source_supplied: bool
    template_source_labels: tuple[str, ...]

    @property
    def template_source_count(self) -> int:
        return len(self.template_source_labels)

    def validation_error(self) -> str | None:
        if self.template_source_count > 1:
            return (
                "selected-operation template sources are mutually exclusive: "
                + ", ".join(self.template_source_labels)
            )
        error_by_state = {
            (False, 1): (
                ", ".join(self.template_source_labels)
                + f" requires {self.selector_source_label}"
            ),
            (True, 0): (
                f"{self.selector_source_label} requires one selected-operation "
                "template source"
            ),
        }
        return error_by_state.get(
            (self.selector_source_supplied, self.template_source_count)
        )


@dataclass(frozen=True)
class SelectedOperationTargetSelectorSourceSelection:
    """Validated target selector source choice for selected-operation commands."""

    source_labels: tuple[str, ...]

    @property
    def supplied(self) -> bool:
        return bool(self.source_labels)

    @property
    def source_count(self) -> int:
        return len(self.source_labels)

    def validation_error(self) -> str | None:
        if self.source_count <= 1:
            return None
        return (
            "selected-operation target selector sources are mutually exclusive: "
            + ", ".join(self.source_labels)
        )


@dataclass(frozen=True)
class CodemodScanQueryMode:
    """Validated family of scan-backed codemod DSL query modes."""

    synthesize_plan: bool
    synthesize_class_plan: bool
    source_index: bool
    selector_path: Path | None
    target_source_selector_path: Path | None
    replacement_plan_selector_path: Path | None
    selected_operation_target_selector_source_labels: tuple[str, ...]
    selected_operation_template_source_labels: tuple[str, ...]
    synthesis_has_registered_detector: bool
    synthesis_execution_requested: bool
    synthesis_authoring_requested: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CodemodScanQueryMode":
        return cls(
            synthesize_plan=args.codemod_synthesize_plan,
            synthesize_class_plan=args.codemod_synthesize_class_plan,
            source_index=args.codemod_source_index,
            selector_path=args.codemod_resolve_selector,
            target_source_selector_path=args.codemod_target_source,
            replacement_plan_selector_path=args.codemod_replacement_plan,
            selected_operation_target_selector_source_labels=(
                SelectedOperationTargetSelectorSource.selected_source_labels(args)
            ),
            selected_operation_template_source_labels=(
                SelectedOperationTemplateSource.selected_source_labels(args)
            ),
            synthesis_has_registered_detector=(
                FindingRecipeSynthesizer.has_registered_detector(
                    args.codemod_goal_detectors
                )
            ),
            synthesis_execution_requested=(
                args.codemod_preflight
                or args.codemod_diff
                or args.codemod_simulate
                or args.codemod_apply
            ),
            synthesis_authoring_requested=(
                args.codemod_synthesis_authoring
                or args.codemod_authoring_bundle_out is not None
            ),
        )

    @property
    def requested(self) -> bool:
        return self.mode_count > 0

    @property
    def needs_analysis(self) -> bool:
        return not self.source_index

    @property
    def needs_source_snapshot(self) -> bool:
        if not self.requested:
            return False
        if not self.synthesize_plan:
            return True
        return (
            self.synthesis_execution_requested
            or self.synthesis_authoring_requested
            or self.synthesis_has_registered_detector
        )

    @property
    def mode_count(self) -> int:
        return sum(
            (
                self.synthesize_plan,
                self.synthesize_class_plan,
                self.source_index,
                self.selector_path is not None,
                self.target_source_selector_path is not None,
                self.replacement_plan_selector_path is not None,
                bool(self.selected_operation_target_selector_source_labels),
            )
        )

    def require_valid(self, parser: argparse.ArgumentParser) -> None:
        target_selector_selection = SelectedOperationTargetSelectorSourceSelection(
            self.selected_operation_target_selector_source_labels
        )
        if (error := target_selector_selection.validation_error()) is not None:
            parser.error(error)
        template_source_selection = SelectedOperationTemplateSourceSelection(
            selector_source_label="selected-operation target selector source",
            selector_source_supplied=target_selector_selection.supplied,
            template_source_labels=self.selected_operation_template_source_labels,
        )
        if (error := template_source_selection.validation_error()) is not None:
            parser.error(error)
        if self.mode_count <= 1:
            return
        parser.error(
            "--codemod-synthesize-plan, --codemod-synthesize-class-plan, "
            "--codemod-source-index, "
            "--codemod-resolve-selector, --codemod-target-source, and "
            "--codemod-replacement-plan, and --codemod-selected-operation-plan "
            "are mutually exclusive"
        )


@dataclass(frozen=True)
class CodemodRecipePlanSourceFile(SourcePathCandidateAuthority):
    """Resolve one explicit recipe source path to a readable Python file."""

    @classmethod
    def from_roots(
        cls,
        requested_path: str,
        roots: tuple[Path, ...],
        cwd: Path,
    ) -> "CodemodRecipePlanSourceFile":
        return cls(
            requested_path=requested_path,
            candidate_paths=cls.candidate_paths_for(requested_path, roots, cwd),
        )

    def source_mapping_entry(self) -> tuple[str, str] | None:
        file_path = self.unique_existing_file_path()
        if file_path is None:
            return None
        return (file_path, Path(file_path).read_text(encoding="utf-8"))

    def unique_existing_file_path(self) -> str | None:
        paths_by_resolved_path: dict[Path, str] = {}
        for file_path in self.candidate_paths:
            path = Path(file_path)
            expanded_path = path.expanduser()
            if expanded_path.is_file():
                paths_by_resolved_path.setdefault(
                    expanded_path.resolve(),
                    expanded_path.as_posix(),
                )
        if len(paths_by_resolved_path) != 1:
            return None
        return tuple(paths_by_resolved_path.values())[0]

    @staticmethod
    def candidate_paths_for(
        requested_path: str,
        roots: tuple[Path, ...],
        cwd: Path,
    ) -> tuple[str, ...]:
        requested = Path(requested_path)
        if requested.is_absolute():
            return (requested.as_posix(),)
        return tuple(
            path.as_posix()
            for path in dict.fromkeys(
                (cwd / requested, *(root / requested for root in roots), requested)
            )
        )


@dataclass(frozen=True)
class CodemodRecipePlanFastSourceSnapshot:
    """Build a narrow source snapshot for exact recipe plans with file paths."""

    sequence: CodemodPlanSequence
    roots: tuple[Path, ...]
    cwd: Path

    def optional_snapshot(self) -> CodemodSourceSnapshot | None:
        if self.sequence.has_unresolved_source_targets:
            return None
        source_by_path = self.source_mapping()
        if source_by_path is None:
            return None
        return CodemodSourceSnapshot.from_source_mapping(source_by_path)

    def source_mapping(self) -> dict[str, str] | None:
        entries = tuple(self.source_mapping_entries())
        if len(entries) != len(self.sequence.explicit_source_paths()):
            return None
        return dict(entries)

    def source_mapping_entries(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            entry
            for requested_path in self.sequence.explicit_source_paths()
            for entry in (
                CodemodRecipePlanSourceFile.from_roots(
                    requested_path=requested_path,
                    roots=self.roots,
                    cwd=self.cwd,
                ).source_mapping_entry(),
            )
            if entry is not None
        )


@dataclass(frozen=True)
class CodemodScanQueryCliCommand(
    CodemodSourceSnapshotRequiredCommand,
    CodemodProjectedFindingReporter,
    ABC,
):
    """Registered command that emits one scan-backed codemod DSL query."""

    source_snapshot_error_message: ClassVar[str] = (
        "codemod scan query requires a source snapshot"
    )
    findings: list[RefactorFinding]
    modules: list[ParsedModule]
    config: DetectorConfig
    roots: tuple[Path, ...]

    @classmethod
    def run_first(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        source_snapshot: CodemodSourceSnapshot | None,
        findings: list[RefactorFinding],
        modules: list[ParsedModule],
        config: DetectorConfig,
        roots: tuple[Path, ...],
    ) -> int | None:
        for command_type in CliCommand.__registry__.values():
            if not issubclass(command_type, cls):
                continue
            command = command_type(
                parser,
                args,
                source_snapshot,
                findings,
                modules,
                config,
                roots,
            )
            if command.requested:
                return command.run()
        return None


class CodemodSynthesisExecutionCliCommand(CodemodScanQueryCliCommand, ABC):
    """Shared execution surface for finding-backed synthesis commands."""

    def with_optional_synthesis_authoring(
        self,
        payload: JsonObject,
        plan: FindingRecipePlan,
    ) -> JsonObject:
        if not self.args.codemod_synthesis_authoring:
            return payload
        return plan.with_authoring_payload(payload)

    @property
    def synthesis_execution_requested(self) -> bool:
        return (
            self.args.codemod_preflight
            or self.args.codemod_diff
            or self.args.codemod_simulate
            or self.args.codemod_apply
        )

    def write_authoring_bundle_if_requested(
        self,
        snapshot: CodemodSourceSnapshot,
        plan: FindingRecipePlan,
    ) -> JsonObject | None:
        output_dir = self.args.codemod_authoring_bundle_out
        if output_dir is None:
            return None
        return CodemodAuthoringBundleWriter(
            output_dir=output_dir,
            snapshot=snapshot,
            plan=plan,
            roots=self.roots,
            cwd=Path.cwd(),
        ).write()

    def apply_synthesized_plan(
        self,
        simulation: "FindingRecipePlanSimulation",
    ) -> bool:
        if not self.args.codemod_apply:
            return False
        simulation.document_simulation.apply()
        return True


class CodemodSynthesizePlanCliCommand(CodemodSynthesisExecutionCliCommand):
    """Emit finding-backed executable codemod recipes."""

    command_id = "codemod_synthesize_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_synthesize_plan

    def run(self) -> int:
        self.require_valid_document_only_mode()
        detector_ids = tuple(self.args.codemod_goal_detectors)
        snapshot = self.source_snapshot
        if snapshot is None:
            finding_recipe_plan = codemod_plan_from_findings(
                self.findings,
                detector_ids=detector_ids,
            )
            authoring_bundle = None
        else:
            finding_recipe_plan = snapshot.plan_from_findings(
                self.findings,
                detector_ids=detector_ids,
            )
            authoring_bundle = self.write_authoring_bundle_if_requested(
                snapshot,
                finding_recipe_plan,
            )
        write_cli_json_artifact(
            self.args.codemod_plan_out,
            finding_recipe_plan.document.to_dict(),
        )
        if self.args.codemod_preflight:
            payload = finding_recipe_plan.preflight_snapshot(snapshot).to_dict()
            payload = self.with_optional_synthesis_authoring(
                payload,
                finding_recipe_plan,
            )
            if authoring_bundle is not None:
                payload["authoring_bundle"] = authoring_bundle
            print(json.dumps(payload, indent=2))
            return CodemodSynthesisExitCodeAuthority(payload["is_clean"]).exit_code()
        if self.synthesis_execution_requested:
            simulation = finding_recipe_plan.simulate_snapshot(snapshot)
            unified_diff = snapshot.unified_diff(simulation.simulation)
            applied = self.apply_synthesized_plan(simulation)
            if self.args.codemod_diff and not self.args.json:
                print(unified_diff, end="")
                return CodemodSynthesisExitCodeAuthority(
                    simulation.is_clean
                ).exit_code()
            payload = {
                **simulation.to_dict(),
                "applied": applied,
                "unified_diff": unified_diff,
            }
            projected_findings = self.optional_projected_finding_report(
                simulation.simulation,
                enabled=self.args.codemod_project_findings,
                expected_removed_finding_ids=(
                    finding_recipe_plan.expected_removed_finding_ids
                ),
            )
            if projected_findings is not None:
                payload["projected_findings"] = projected_findings.to_dict()
                self.write_continuation_plan_if_requested(projected_findings)
            payload = self.with_optional_synthesis_authoring(
                payload,
                finding_recipe_plan,
            )
            if authoring_bundle is not None:
                payload["authoring_bundle"] = authoring_bundle
            print(json.dumps(payload, indent=2))
            return CodemodSynthesisExitCodeAuthority(simulation.is_clean).exit_code()
        if self.args.codemod_synthesize_document_only:
            payload = finding_recipe_plan.document.to_dict()
        else:
            payload = finding_recipe_plan.to_dict()
            payload = self.with_optional_synthesis_authoring(
                payload,
                finding_recipe_plan,
            )
        if authoring_bundle is not None:
            payload["authoring_bundle"] = authoring_bundle
        print(json.dumps(payload, indent=2))
        return 0

    def require_valid_document_only_mode(self) -> None:
        if not self.args.codemod_synthesize_document_only:
            return
        if self.synthesis_execution_requested:
            self.parser.error(
                "--codemod-synthesize-document-only cannot be combined with "
                "--codemod-preflight, --codemod-diff, --codemod-simulate, "
                "or --codemod-apply"
            )
        if self.args.codemod_synthesis_authoring:
            self.parser.error(
                "--codemod-synthesis-authoring cannot be combined with "
                "--codemod-synthesize-document-only"
            )


class CodemodSynthesizeClassPlanCliCommand(CodemodSynthesisExecutionCliCommand):
    """Emit graph-clustered finding-backed codemod plans with scaffolds."""

    command_id = "codemod_synthesize_class_plan"

    @property
    def requested(self) -> bool:
        return self.args.codemod_synthesize_class_plan

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        report = codemod_class_plan_from_findings(
            self.findings,
            root=self.roots[0],
            selector_context=snapshot,
            detector_ids=tuple(self.args.codemod_goal_detectors),
        )
        write_cli_json_artifact(
            self.args.codemod_plan_out,
            report.finding_plan.document.to_dict(),
        )
        if self.args.codemod_preflight:
            preflight = report.finding_plan.preflight_snapshot(snapshot)
            payload = {
                **report.to_dict(),
                "preflight_report": preflight.preflight_report.to_dict(),
                "is_clean": preflight.is_clean,
            }
            payload = self.with_optional_synthesis_authoring(
                payload,
                report.finding_plan,
            )
            print(json.dumps(payload, indent=2))
            return CodemodSynthesisExitCodeAuthority(payload["is_clean"]).exit_code()
        if self.synthesis_execution_requested:
            simulation = report.finding_plan.simulate_snapshot(snapshot)
            unified_diff = snapshot.unified_diff(simulation.simulation)
            applied = self.apply_synthesized_plan(simulation)
            if self.args.codemod_diff and not self.args.json:
                print(unified_diff, end="")
                return CodemodSynthesisExitCodeAuthority(
                    simulation.is_clean
                ).exit_code()
            payload = {
                **report.to_dict(),
                "simulation_result": simulation.to_dict(),
                "applied": applied,
                "unified_diff": unified_diff,
            }
            projected_findings = self.optional_projected_finding_report(
                simulation.simulation,
                enabled=self.args.codemod_project_findings,
                expected_removed_finding_ids=(
                    report.finding_plan.expected_removed_finding_ids
                ),
            )
            if projected_findings is not None:
                payload["projected_findings"] = projected_findings.to_dict()
                payload["class_plan_projected_deltas"] = (
                    projected_findings.class_plan_delta_report(report).to_dict()
                )
                self.write_continuation_plan_if_requested(projected_findings)
            payload = self.with_optional_synthesis_authoring(
                payload,
                report.finding_plan,
            )
            print(json.dumps(payload, indent=2))
            return CodemodSynthesisExitCodeAuthority(simulation.is_clean).exit_code()
        payload = report.to_dict()
        payload = self.with_optional_synthesis_authoring(payload, report.finding_plan)
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
    writes_plan_document: ClassVar[bool] = False

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        try:
            selector = load_codemod_target_selector(self.selector_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        payload = self.payload_for_selector(snapshot, selector)
        if self.writes_plan_document:
            write_cli_json_artifact(
                self.args.codemod_plan_out,
                JsonObject(payload["document"]),
            )
        print(
            json.dumps(
                payload,
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
    writes_plan_document = True
    payload_builder = CodemodSelectorPayloadBuilder(
        CodemodSourceSnapshot.replacement_plan_scaffold_report
    )

    @property
    def requested(self) -> bool:
        return self.args.codemod_replacement_plan is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_replacement_plan


class SelectedOperationPlanCliMode(ABC, metaclass=AutoRegisterMeta):
    """Registered execution mode for selected-operation plan scaffolds."""

    __registry__: ClassVar[dict[str, type["SelectedOperationPlanCliMode"]]] = {}
    __registry_key__ = "mode_id"
    __skip_if_no_key__ = True

    mode_id: ClassVar[str]
    registry_order: ClassVar[int]

    @classmethod
    def resolve(
        cls,
        execution_mode: CodemodExecutionMode,
    ) -> "SelectedOperationPlanCliMode":
        for mode_type in cls.ordered_mode_types():
            mode = mode_type()
            if mode.matches(execution_mode):
                return mode
        raise RuntimeError("selected-operation plan CLI mode registry is empty")

    @classmethod
    def ordered_mode_types(cls) -> tuple[type["SelectedOperationPlanCliMode"], ...]:
        return tuple(
            sorted(
                cls.__registry__.values(),
                key=lambda mode_type: mode_type.registry_order,
            )
        )

    @abstractmethod
    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        raise NotImplementedError


class SelectedOperationPlanPreflightCliMode(SelectedOperationPlanCliMode):
    """Preflight a generated selected-operation plan document."""

    mode_id = "preflight"
    registry_order = 10

    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        return execution_mode.preflight

    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        return command.emit_preflight_report(snapshot, scaffold)


class SelectedOperationPlanDiffCliMode(SelectedOperationPlanCliMode):
    """Diff a generated selected-operation plan document."""

    mode_id = "diff"
    registry_order = 20

    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        return execution_mode.diff

    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        simulation = command.simulation_for_scaffold(snapshot, scaffold)
        unified_diff = snapshot.unified_diff(simulation.simulation)
        if command.args.json:
            return command.emit_simulation_report(
                scaffold,
                simulation,
                applied=False,
                unified_diff=unified_diff,
            )
        command.write_scaffold_plan_if_requested(scaffold)
        print(unified_diff, end="")
        return CodemodSynthesisExitCodeAuthority(simulation.is_clean).exit_code()


class SelectedOperationPlanSimulateCliMode(SelectedOperationPlanCliMode):
    """Simulate a generated selected-operation plan document."""

    mode_id = "simulate"
    registry_order = 30

    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        return execution_mode.simulate

    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        simulation = command.simulation_for_scaffold(snapshot, scaffold)
        return command.emit_simulation_report(
            scaffold,
            simulation,
            applied=False,
            unified_diff=snapshot.unified_diff(simulation.simulation),
        )


class SelectedOperationPlanApplyCliMode(SelectedOperationPlanCliMode):
    """Apply a generated selected-operation plan document."""

    mode_id = "apply"
    registry_order = 40

    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        return execution_mode.apply

    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        simulation = command.simulation_for_scaffold(snapshot, scaffold)
        simulation.apply()
        return command.emit_simulation_report(
            scaffold,
            simulation,
            applied=True,
            unified_diff=None,
        )


class SelectedOperationPlanScaffoldCliMode(SelectedOperationPlanCliMode):
    """Emit the generated selected-operation plan scaffold."""

    mode_id = "scaffold"
    registry_order = 50

    def matches(self, execution_mode: CodemodExecutionMode) -> bool:
        return not execution_mode.requested

    def run(
        self,
        command: "CodemodSelectedOperationPlanCliCommand",
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        del snapshot
        return command.emit_scaffold(scaffold)


class CodemodSelectedOperationPlanCliCommand(CodemodScanQueryCliCommand):
    """Emit an apply-selected-targets plan for selector and operation templates."""

    command_id = "codemod_selected_operation_plan"

    @property
    def requested(self) -> bool:
        return bool(SelectedOperationTargetSelectorSource.selected_sources(self.args))

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        selector = SelectedOperationTargetSelectorSource.required_source(
            self.args,
            self.parser,
        ).target_selector(self.args, self.parser)
        scaffold = self.scaffold_for_selector(snapshot, selector)
        execution_mode = CodemodExecutionMode.from_namespace(self.args)
        mode = SelectedOperationPlanCliMode.resolve(execution_mode)
        return mode.run(self, snapshot, scaffold)

    def emit_scaffold(
        self,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        self.write_scaffold_plan_if_requested(scaffold)
        print(json.dumps(scaffold.to_dict(), indent=2))
        return 0

    def write_scaffold_plan_if_requested(
        self,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> None:
        write_cli_json_artifact(self.args.codemod_plan_out, scaffold.document.to_dict())

    def scaffold_for_selector(
        self,
        snapshot: CodemodSourceSnapshot,
        selector: CodemodTargetSelector,
    ) -> CodemodSelectedOperationPlanScaffoldReport:
        operation_plan_template = SelectedOperationTemplateSource.required_source(
            self.args,
            self.parser,
        ).operation_plan_template(self.args, self.parser)
        return snapshot.selected_operation_plan_scaffold_report(
            selector,
            operation_plan_template,
        )

    def emit_preflight_report(
        self,
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> int:
        report = scaffold.document.preflight_snapshot(snapshot)
        self.write_scaffold_plan_if_requested(scaffold)
        print(
            json.dumps(
                {
                    **CodemodPlanPreflightPayload(report).to_dict(),
                    "scaffold": scaffold.to_dict(),
                    "document": scaffold.document.to_dict(),
                },
                indent=2,
            )
        )
        if report.is_clean:
            return 0
        return 1

    def simulation_for_scaffold(
        self,
        snapshot: CodemodSourceSnapshot,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
    ) -> CodemodPlanDocumentSimulation:
        try:
            return scaffold.document.simulate_snapshot(snapshot)
        except CodemodOperationPreflightError as error:
            self.emit_preflight_failure(scaffold, error)
            raise SystemExit(1) from error

    def emit_preflight_failure(
        self,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
        error: CodemodOperationPreflightError,
    ) -> None:
        print(
            json.dumps(
                {
                    **CodemodPreflightFailurePayload(error.report).to_dict(),
                    "scaffold": scaffold.to_dict(),
                    "document": scaffold.document.to_dict(),
                },
                indent=2,
            )
        )

    def emit_simulation_report(
        self,
        scaffold: CodemodSelectedOperationPlanScaffoldReport,
        simulation: CodemodPlanDocumentSimulation,
        *,
        applied: bool,
        unified_diff: str | None,
    ) -> int:
        payload = CodemodSimulationPayload(
            simulation.simulation,
            applied=applied,
            post_guard_report=simulation.architecture_guard_report,
            unified_diff=unified_diff,
        ).to_dict()
        projected_findings = self.optional_projected_finding_report(
            simulation.simulation,
            enabled=self.args.codemod_project_findings,
        )
        if projected_findings is not None:
            payload["projected_findings"] = projected_findings.to_dict()
            self.write_continuation_plan_if_requested(projected_findings)
        payload["scaffold"] = scaffold.to_dict()
        payload["document"] = scaffold.document.to_dict()
        self.write_scaffold_plan_if_requested(scaffold)
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return CodemodSynthesisExitCodeAuthority(simulation.is_clean).exit_code()


def main() -> int:
    """Run the command-line interface and return a process status code."""
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for nominal architecture."
    )
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)
    args = parser.parse_args()

    if args.codemod_plan_out is not None and not codemod_plan_output_supported(args):
        parser.error("--codemod-plan-out requires a plan-producing codemod command")
    if args.codemod_authoring_bundle_out is not None and not (
        args.codemod_synthesize_plan and args.codemod_synthesis_authoring
    ):
        parser.error(
            "--codemod-authoring-bundle-out requires "
            "--codemod-synthesize-plan --codemod-synthesis-authoring"
        )

    early_exit_code = CliEarlyExitCommand.run_first(parser, args)
    if early_exit_code is not None:
        return early_exit_code

    config = DetectorConfig.from_namespace(args)
    codemod_scan_query_mode = CodemodScanQueryMode.from_namespace(args)
    codemod_scan_query_mode.require_valid(parser)
    try:
        json_payload_profile = JsonPayloadProfile.from_cli_value(args.json_payload)
    except ValueError as error:
        parser.error(str(error))
    explicit_impact_ranking_request = args.include_impact_ranking is not None
    impact_ranking_policy = JsonPayloadImpactRankingPolicy(
        explicit_request=args.include_impact_ranking,
        json_enabled=args.json,
        payload_profile=json_payload_profile,
    )
    args.include_impact_ranking = impact_ranking_policy.include_impact_ranking
    JsonDocumentInputSet.from_option_paths(
        (
            ("--codemod-plan", (args.codemod_plan,)),
            ("--codemod-workflow-plan", (args.codemod_workflow_plan,)),
            ("--codemod-resolve-selector", (args.codemod_resolve_selector,)),
            ("--codemod-target-source", (args.codemod_target_source,)),
            ("--codemod-replacement-plan", (args.codemod_replacement_plan,)),
            (
                "--codemod-selected-operation-plan",
                (args.codemod_selected_operation_plan,),
            ),
            ("--codemod-operation-template", (args.codemod_operation_template,)),
        )
    ).require_at_most_one_stdin(parser)
    codemod_execution_mode = CodemodExecutionMode.from_namespace(args)
    codemod_execution_mode.require_valid(parser)
    codemod_requested = (
        args.codemod_plan is not None
        or codemod_execution_mode.requested
        or args.codemod_fixpoint
        or args.codemod_workflow_plan is not None
        or args.codemod_refactor_goal is not None
        or codemod_scan_query_mode.requested
    )
    if args.codemod_synthesis_authoring and not args.codemod_synthesize_plan:
        parser.error("--codemod-synthesis-authoring requires --codemod-synthesize-plan")
    if (
        args.codemod_continuation_plan_out is not None
        and not args.codemod_project_findings
    ):
        parser.error(
            "--codemod-continuation-plan-out requires --codemod-project-findings"
        )
    if args.codemod_workflow_plan is not None and (
        args.codemod_fixpoint or args.codemod_refactor_goal is not None
    ):
        parser.error(
            "--codemod-workflow-plan cannot be combined with "
            "--codemod-fixpoint or --codemod-refactor-goal"
        )
    if args.codemod_fixpoint_plan_out is not None and not args.codemod_fixpoint:
        parser.error("--codemod-fixpoint-plan-out requires --codemod-fixpoint")
    if args.codemod_goal_plan_out is not None and args.codemod_refactor_goal is None:
        parser.error("--codemod-goal-plan-out requires --codemod-refactor-goal")
    codemod_workflow_plan = (
        load_codemod_workflow_plan(args.codemod_workflow_plan)
        if args.codemod_workflow_plan is not None
        else None
    )
    codemod_plan_sequence = (
        load_codemod_plan_sequence(args.codemod_plan)
        if args.codemod_plan is not None
        else CodemodPlanSequence()
    )
    codemod_execution_request = CodemodPlanExecutionRequest(
        sequence=codemod_plan_sequence,
        mode=codemod_execution_mode,
    )
    if (
        codemod_requested
        and not explicit_impact_ranking_request
        and not args.codemod_fixpoint
        and codemod_workflow_plan is None
        and args.codemod_refactor_goal is None
        and not codemod_scan_query_mode.requested
        and not codemod_plan_sequence.has_recipes
    ):
        args.include_impact_ranking = True
    if args.codemod_fixpoint and args.codemod_fixpoint_max_iterations < 1:
        parser.error("--codemod-fixpoint-max-iterations must be at least 1")
    if (
        codemod_requested
        and not args.codemod_fixpoint
        and codemod_workflow_plan is None
        and args.codemod_refactor_goal is None
        and not codemod_scan_query_mode.requested
        and not args.include_impact_ranking
        and not codemod_plan_sequence.has_recipes
    ):
        parser.error("--codemod-* options require impact ranking or recipe rewrites")
    if (
        codemod_requested
        and not args.include_impact_ranking
        and codemod_plan_sequence.has_authority_boundaries
    ):
        parser.error("authority-boundary codemod plans require impact ranking")
    if codemod_requested and args.import_lean_export is not None:
        parser.error("--codemod-* options require parsed Python source paths")

    if args.calibrate is not None:
        parse_cache_dir = ParseCacheDirAuthority(
            root=args.calibrate.parent,
            requested_parse_cache_dir=args.cache_dir,
            use_parse_cache=args.use_parse_cache,
        ).parse_cache_dir()
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

    requested_roots = tuple(Path(path) for path in args.paths)
    path_scope = AnalysisPathScope.from_requested_roots(
        requested_roots,
        tuple(args.context_roots),
        auto_context=args.auto_context_root,
    )
    roots = path_scope.analysis_roots
    root = path_scope.primary_analysis_root
    parse_cache_dir = ParseCacheDirAuthority(
        root=root,
        requested_parse_cache_dir=args.cache_dir,
        use_parse_cache=args.use_parse_cache,
    ).parse_cache_dir()
    analysis_cache_dir = analysis_cache_dir_for_root(
        root,
        parse_cache_dir,
        args.use_parse_cache,
    )
    source_policy = PythonSourcePathPolicy(include_tests=args.include_tests)
    semantic_descent_cache_dir = SemanticDescentCacheDirAuthority(
        parse_cache_dir=parse_cache_dir,
        use_parse_cache=args.use_parse_cache,
    ).cache_dir()
    semantic_descent_analysis_source = SemanticDescentGraphAnalysisSource(
        cache_dir=semantic_descent_cache_dir,
        cache_roots=roots,
        source_policy=source_policy,
        use_cache=args.use_parse_cache,
    )
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

    emitted_semantic_refactor_gate = (
        args.json and json_payload_profile.sections.semantic_refactor_gate
    ) or args.plans_only
    try:
        SemanticRefactorGateMode.from_flags(
            include_impact_ranking=args.include_impact_ranking
            or args.codemod_fixpoint
            or codemod_workflow_plan is not None
            or codemod_scan_query_mode.requested,
            semantic_refactor_gate=emitted_semantic_refactor_gate,
            raw_findings=(
                args.raw_findings
                or impact_ranking_policy.lightweight_profile_acknowledges_raw_findings
            ),
        ).require_authority_boundary_mode()
    except SemanticRefactorGateModeError as error:
        parser.error(str(error))

    fast_codemod_source_snapshot = None
    if (
        codemod_execution_request.exact_recipe_execution
        and not args.include_impact_ranking
        and not args.codemod_fixpoint
        and codemod_workflow_plan is None
        and args.codemod_refactor_goal is None
        and args.import_lean_export is None
        and not codemod_scan_query_mode.requested
    ):
        fast_codemod_source_snapshot = CodemodRecipePlanFastSourceSnapshot(
            sequence=codemod_execution_request.sequence,
            roots=roots,
            cwd=Path.cwd(),
        ).optional_snapshot()
    if fast_codemod_source_snapshot is not None:
        fast_codemod_execution_result = CodemodCliExecution(
            parser=parser,
            args=args,
            source_snapshot=fast_codemod_source_snapshot,
            impact_candidates=None,
            execution_request=codemod_execution_request,
            architecture_guard_evaluator=ArchitectureGuardSourceEvaluator([], ()),
            modules=[],
            findings=[],
            config=config,
            roots=roots,
        ).run()
        if fast_codemod_execution_result is not None:
            return fast_codemod_execution_result

    if args.import_lean_export is None:
        preparse_cache_policy = JsonSummaryPreparseCachePolicy(
            json_enabled=args.json,
            payload_profile=json_payload_profile,
            load_bearing_ranking_enabled=args.include_impact_ranking,
            parsed_modules_required=(
                args.codemod_fixpoint
                or args.codemod_refactor_goal is not None
                or codemod_plan_sequence.has_recipes
                or codemod_scan_query_mode.needs_source_snapshot
            ),
            analysis_cache_dir=analysis_cache_dir,
            focused_report_filter=path_scope.has_report_filter,
        )
        fast_cache_result = None
        cached_semantic_descent_graph = None
        preparse_cache_mode = preparse_cache_policy.mode
        if (
            codemod_scan_query_mode.needs_analysis
            and preparse_cache_policy.cache_lookup_enabled
            and preparse_cache_mode.enabled
        ):
            started = perf_counter()
            fast_semantic_descent_context = (
                FastPreparseSemanticDescentSourceAuthority(
                    preparse_cache_policy=preparse_cache_policy,
                    base_source=semantic_descent_analysis_source,
                    roots=roots,
                    semantic_descent_cache_dir=semantic_descent_cache_dir,
                    source_policy=source_policy,
                    use_cache=args.use_parse_cache,
                ).context()
            )
            latest_semantic_descent_graph = fast_semantic_descent_context.latest_graph
            fast_semantic_descent_analysis_source = (
                fast_semantic_descent_context.analysis_source
            )
            fast_cache_request = CachedPathAnalysisRequest(
                roots=roots,
                config=config,
                parse_cache_dir=parse_cache_dir,
                use_parse_cache=args.use_parse_cache,
                parse_workers=args.parse_workers,
                analysis_workers=args.analysis_workers,
                source_policy=source_policy,
                reuse_policy=preparse_cache_mode.reuse_policy(
                    focused_report_filter=path_scope.has_report_filter
                ),
                semantic_descent_source=fast_semantic_descent_analysis_source,
            )
            fast_cache_authority = FastCachedPathAnalysisAuthority(fast_cache_request)
            if (
                json_payload_profile is JsonPayloadProfile.loop
                and not path_scope.has_report_filter
            ):
                summary_cache_result = fast_cache_authority.summary_result()
                if summary_cache_result is not None:
                    analysis_seconds = round(perf_counter() - started, 3)
                    timing = ScanTiming(
                        parse_seconds=0.0,
                        analysis_seconds=analysis_seconds,
                        analysis_cache_status=AnalysisCacheStatus.HIT,
                    )
                    print(
                        json.dumps(
                            JsonLoopCachePayloadBuilder(
                                summary_cache_result,
                                timing,
                            ).to_dict(),
                            indent=2,
                        )
                    )
                    return 0
            fast_cache_result = fast_cache_authority.result()
            if (
                fast_cache_result is not None
                and preparse_cache_mode.requires_semantic_descent_cache
            ):
                cached_semantic_descent_graph = (
                    load_cached_semantic_descent_graph_for_roots(
                        roots,
                        cache_dir=semantic_descent_cache_dir,
                        source_policy=source_policy,
                    )
                )
                if (
                    cached_semantic_descent_graph is None
                    and preparse_cache_policy.uses_evidence_local_partial_reuse
                ):
                    cached_semantic_descent_graph = latest_semantic_descent_graph
                if cached_semantic_descent_graph is None:
                    fast_cache_result = None
            fast_cache_seconds = round(perf_counter() - started, 3)
        if fast_cache_result is not None:
            modules = []
            parse_seconds = 0.0
            analysis_cache_status = fast_cache_result.cache_status
            findings = path_scope.filter_findings(fast_cache_result.findings)
            analysis_seconds = fast_cache_seconds
        else:
            started = perf_counter()
            modules = parse_python_module_roots(
                roots,
                cache_dir=parse_cache_dir,
                use_parse_cache=args.use_parse_cache,
                parse_workers=args.parse_workers,
                source_policy=source_policy,
            )
            parse_seconds = round(perf_counter() - started, 3)
            if not codemod_scan_query_mode.needs_analysis:
                findings = []
                analysis_seconds = 0.0
                analysis_cache_status = None
            else:
                started = perf_counter()
                analysis_result = analyze_modules_with_cache(
                    roots,
                    modules,
                    config,
                    analysis_cache_dir=analysis_cache_dir,
                    analysis_workers=args.analysis_workers,
                    source_policy=source_policy,
                    semantic_descent_source=semantic_descent_analysis_source,
                )
                unfiltered_findings = analysis_result.findings
                analysis_cache_status = analysis_result.cache_status
                findings = path_scope.filter_findings(unfiltered_findings)
                analysis_seconds = round(perf_counter() - started, 3)
    else:
        modules = []
        findings = analyze_lean_export(args.import_lean_export)
        parse_seconds = 0.0
        analysis_seconds = 0.0
        analysis_cache_status = None
        cached_semantic_descent_graph = None
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
    authority_boundary_plans = codemod_plan_sequence.authority_boundaries
    architecture_guard_rules = codemod_plan_sequence.guard_suite.to_tuple()
    architecture_guard_evaluator = ArchitectureGuardSourceEvaluator(
        modules,
        architecture_guard_rules,
    )
    if codemod_workflow_plan is not None:
        report = codemod_workflow_plan.run(
            resolved_dir=parse_cache_dir,
            enabled=args.use_parse_cache,
            roots=roots,
            config=config,
            parse_workers=args.parse_workers,
            guard_suite=codemod_plan_sequence.guard_suite,
            dry_run=not args.codemod_apply,
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=findings,
            ),
        )
        write_cli_json_artifact(args.codemod_plan_out, report.replay_sequence.to_dict())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.to_markdown())
        return 0 if report.completed else 1
    if args.codemod_fixpoint:
        report = CodemodFixpointRunner(
            resolved_dir=parse_cache_dir,
            enabled=args.use_parse_cache,
            roots=roots,
            config=config,
            parse_workers=args.parse_workers,
            max_iterations=args.codemod_fixpoint_max_iterations,
            guard_suite=codemod_plan_sequence.guard_suite,
            dry_run=not args.codemod_apply,
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=findings,
            ),
        ).run()
        replay_plan_payload = report.replay_plan.sequence.to_dict()
        write_cli_json_artifact(args.codemod_fixpoint_plan_out, replay_plan_payload)
        write_cli_json_artifact(args.codemod_plan_out, replay_plan_payload)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_codemod_fixpoint_markdown(report))
        return 0 if report.completed else 1
    if args.codemod_refactor_goal is not None:
        try:
            refactor_goal = codemod_refactor_goal_from_args(args)
        except ValueError as error:
            parser.error(str(error))
        report = CodemodRefactorGoalRunner(
            resolved_dir=parse_cache_dir,
            enabled=args.use_parse_cache,
            roots=roots,
            config=config,
            parse_workers=args.parse_workers,
            goal=refactor_goal,
            guard_suite=codemod_plan_sequence.guard_suite,
            dry_run=not args.codemod_apply,
            initial_scan=CodemodFixpointScan(
                modules=modules,
                findings=findings,
            ),
        ).run()
        replay_plan_payload = report.replay_sequence.to_dict()
        write_cli_json_artifact(args.codemod_goal_plan_out, replay_plan_payload)
        write_cli_json_artifact(args.codemod_plan_out, replay_plan_payload)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_codemod_refactor_goal_markdown(report))
        return 0 if report.completed else 1
    impact_ranking = None
    architecture_guard_report = None
    source_snapshot = None
    if (
        args.include_impact_ranking
        or codemod_plan_sequence.has_recipes
        or codemod_scan_query_mode.needs_source_snapshot
    ):
        started = perf_counter()
        source_snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
        source_index_seconds = round(perf_counter() - started, 3)

    scan_query_result = CodemodScanQueryCliCommand.run_first(
        parser,
        args,
        source_snapshot,
        findings,
        modules,
        config,
        roots,
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
        if not codemod_plan_sequence.has_recipes:
            source_snapshot = None
    timing = ScanTiming(
        parse_seconds=parse_seconds,
        analysis_seconds=analysis_seconds,
        planning_seconds=planning_seconds,
        source_index_seconds=source_index_seconds,
        analysis_cache_status=analysis_cache_status,
    )

    codemod_execution_result = CodemodCliExecution(
        parser=parser,
        args=args,
        source_snapshot=source_snapshot,
        impact_candidates=codemod_candidates,
        execution_request=codemod_execution_request,
        architecture_guard_evaluator=architecture_guard_evaluator,
        modules=modules,
        findings=findings,
        config=config,
        roots=roots,
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
                    semantic_descent_source=JsonSemanticDescentPayloadSource(
                        modules=modules,
                        parse_cache_dir=parse_cache_dir,
                        cached_repository_graph=cached_semantic_descent_graph,
                        use_parse_cache=args.use_parse_cache,
                    ),
                    payload_sections=json_payload_profile.sections,
                    raw_findings=args.raw_findings,
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
