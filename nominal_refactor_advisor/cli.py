"""CLI and top-level analysis helpers.

This module contains the programmatic entrypoints used by tests and automation as
well as the command-line interface used by developers. The public helpers are the
recommended way to analyse a path or inspect subsystem evidence clusters.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Self, TypeAlias, cast

from metaclass_registry import AutoRegisterMeta

from .analysis import (
    AnalysisPathScope,
    CachedPathAnalysisRequest,
    DetectorTypePartition,
    EvidenceLocalPartialDetectorSelection,
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphCacheContext,
    SemanticDescentGraphAnalysisSource,
    analysis_cache_dir_for_root,
    analyze_compact_roots_with_cache,
    analyze_lean_export,
    analyze_module_detector_types_with_cache,
    analyze_modules_with_cache,
    analyze_path,  # noqa: F401 - re-exported by nominal_refactor_advisor.__init__
    analyze_paths,  # noqa: F401 - re-exported by nominal_refactor_advisor.__init__
    default_detector_types_for_analysis,
    plan_path,  # noqa: F401 - re-exported by nominal_refactor_advisor.__init__
    plan_paths,  # noqa: F401 - re-exported by nominal_refactor_advisor.__init__
    release_module_analysis_memory,
    SortedFindingsAuthority,
)
from .analysis_cache import (
    AnalysisCacheStatus,
    AnalysisExecutionPlanCacheIdentity,
    AnalysisFindingCache,
)
from .ast_tools import ParsedModule, PythonSourcePathPolicy, parse_python_module_roots
from .cache_paths import (
    ParseCachePolicy,
    default_parse_cache_dir,
    maintain_default_cache,
)
from .calibration import (
    CalibrationReport,
    format_calibration_markdown,
    run_calibration_manifest,
)
from .codemod import (
    ArchitectureGuardReport,
    ArchitectureGuardRule,
    CodemodJsonReport,
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
    CodemodPlanDocument,
    CodemodPlanPreflightReport,
    CodemodPlanRoot,
    CodemodPlanSequence,
    CodemodPlanSequenceSimulation,
    CodemodTargetSelector,
    CodemodSimulationReport,
    CodemodSourceContext,
    CodemodSourceSnapshot,
    FindingRecipeClassPlanReport,
    FindingRecipePlan,
    FindingRecipePlanPreflight,
    FindingRecipePlanSimulation,
    FindingRecipeFrontierBudget,
    JsonObject,
    JsonValue,
    RefactorConcept,
    SourcePathCandidateAuthority,
    SourcePathCandidateSet,
    codemod_class_plan_from_findings,
    evaluate_architecture_guards,
    module_name_from_source_path,
)
from .codemod_source_cache import CodemodSourceContextCache
from .codemod_workflow import (
    CodemodWorkflowScan,
    CodemodProjectedFindingReport,
    CodemodRefactorGoalReport,
    CodemodRefactorGoalRunner,
    CodemodRefactorTrajectoryBudget,
    CodemodSimulationFindingProjection,
)
from .detectors import DetectorConfig, IssueDetector
from .deadline import ScanDeadline, ScanDeadlineExceeded, enforce_scan_deadline
from .economics import (
    EconomicsProofReport,
    LineChangeBudget,
    RefactorEvidenceEconomics,
    RepositoryChangeBudget,
    ScanEconomicsProof,
    build_economics_proof_report,
)
from .finding_counts import FindingSummary
from .structural_overlap import (
    StructuralOverlapReport,
    StructuralOverlapReportLimits,
    build_structural_overlap_report,
)
from .models import RefactorFinding, RefactorPlan
from .observation_graph import build_observation_graph
from .patterns import PatternId
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
    SemanticRefactorGateReport,
    ssot_authority_findings,
)
from .semantic_descent import (
    SemanticDescentGraph,
    SemanticDescentGraphPayloadReport,
    build_finding_backed_semantic_descent_graph,
)
from .source_index import build_source_index

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
CodemodSelectorReportFactory: TypeAlias = Callable[
    [CodemodSourceSnapshot, CodemodTargetSelector],
    CodemodJsonReport,
]


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
                "evidence surface."
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
                "AST parse cache directory. Defaults to an NRA cache-home entry "
                "keyed by the analysis root."
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
            help="Also emit non-actionable subsystem structural hypotheses.",
        ),
        CliArgumentSpec(
            flags=("--include-execution-plan",),
            action="store_true",
            help=(
                "Also emit graph-grounded classes derived from shared structural "
                "evidence."
            ),
        ),
        CliArgumentSpec(
            flags=("--plans-only",),
            action="store_true",
            help="Emit only subsystem-level structural hypotheses.",
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
            flags=("--include-structural-overlap",),
            action="store_true",
            dest="include_structural_overlap",
            default=None,
            default_supplied=True,
            help=(
                "Report non-actionable structural-overlap evidence for groups of "
                "findings and source targets."
            ),
        ),
        CliArgumentSpec(
            flags=("--no-structural-overlap",),
            action="store_false",
            dest="include_structural_overlap",
            default=None,
            default_supplied=True,
            help=("Skip the non-actionable structural-overlap evidence report."),
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
            flags=("--structural-overlap-max-groups",),
            value_type=int,
            default=25,
            help="Maximum structural-overlap evidence groups to report.",
        ),
        CliArgumentSpec(
            flags=("--structural-overlap-min-findings",),
            value_type=int,
            default=2,
            help="Minimum findings a structural-overlap group must contain.",
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
                "--codemod-simulate or --codemod-apply to "
                "execute the synthesized batch in the same scan."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-synthesize-class-plan",),
            action="store_true",
            help=(
                "Scan paths, cluster findings into graph-derived refactor classes, "
                "and emit the executable typed DSL plan for each class."
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
            flags=("--codemod-source-index",),
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
            flags=("--codemod-project-source-index",),
            action="store_true",
            help=(
                "With --codemod-project-findings, include the simulated "
                "source-index payload in JSON output."
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
            flags=("--codemod-refactor-goal",),
            value_type=str,
            help=(
                "Prove dependent DSL stages in memory and, with --codemod-apply, "
                "commit the completed sequence as one revision-checked transaction. "
                "Supported goals: "
                + ", ".join(
                    concept_type.concept_key()
                    for concept_type in RefactorConcept.declaration_types()
                )
                + "."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-detector",),
            action="append",
            dest="codemod_goal_detectors",
            default=[],
            help=(
                "Restrict one-shot plan synthesis to findings from this detector "
                "(can be repeated)."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-max-stages",),
            value_type=int,
            default=8,
            help=(
                "Maximum reachable trajectory depth proved for --codemod-refactor-goal."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-max-states",),
            value_type=int,
            default=512,
            help=(
                "Maximum exact source states exhaustively explored for "
                "--codemod-refactor-goal."
            ),
        ),
        CliArgumentSpec(
            flags=("--codemod-goal-max-branches",),
            value_type=int,
            default=256,
            help=(
                "Maximum compatible recipe batches enumerated at each exact "
                "source state for --codemod-refactor-goal."
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

    summary: FindingSummary

    def to_dict(self) -> JsonObject:
        return {
            "by_pattern": tuple(
                {
                    "pattern_id": pattern_id,
                    "pattern_name": PatternId(pattern_id).display_name,
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
    semantic_boundary_evidence = "semantic_boundary_evidence"


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


def _full_execution_plan_payload(report: RefactorExecutionPlanReport) -> JsonObject:
    return report.to_dict()


def _count_only_execution_plan_payload(
    report: RefactorExecutionPlanReport,
) -> JsonObject:
    return {
        **report.to_dict(),
        "edges": (),
        "edge_payload_mode": "count_only",
        "edge_count": len(report.edges),
    }


class JsonExecutionPlanPayloadMode(Enum):
    """Declaration-owned execution-plan projection for one JSON profile."""

    FULL = ("full", _full_execution_plan_payload)
    COUNT_ONLY = ("count_only", _count_only_execution_plan_payload)

    def __new__(
        cls,
        value: str,
        payload_builder: Callable[[RefactorExecutionPlanReport], JsonObject],
    ) -> "JsonExecutionPlanPayloadMode":
        member = object.__new__(cls)
        member._value_ = value
        member._payload_builder = payload_builder
        return member

    def payload(
        self,
        report: RefactorExecutionPlanReport,
    ) -> JsonObject:
        return self._payload_builder(report)


@dataclass(frozen=True)
class JsonPayloadSections:
    """Declared section policy for one JSON payload profile."""

    finding_payload_mode: JsonFindingPayloadMode = JsonFindingPayloadMode.full
    source_index: bool = True
    observation_graph: bool = True
    observation_fibers: bool = True
    semantic_descent_graph: bool = True
    semantic_refactor_gate: bool = True
    finding_recipe_plan: bool = True
    payload_timing: bool = False
    default_structural_overlap: bool = False
    execution_plan_payload_mode: JsonExecutionPlanPayloadMode = (
        JsonExecutionPlanPayloadMode.FULL
    )

    @property
    def needs_observation_graph(self) -> bool:
        return self.observation_graph or self.observation_fibers

    @property
    def lightweight_status_payload(self) -> bool:
        return (
            not self.source_index
            and not self.needs_observation_graph
            and not self.semantic_descent_graph
            and not self.semantic_refactor_gate
            and not self.finding_recipe_plan
            and not self.default_structural_overlap
        )

    @property
    def compact_analysis_compatible(self) -> bool:
        """Whether every requested section has an AST-free production source."""

        return (
            not self.source_index
            and not self.needs_observation_graph
            and not self.finding_recipe_plan
        )


@dataclass(frozen=True)
class JsonPayloadSourceSnapshotDemand:
    """Source-snapshot demand induced by one payload section policy."""

    sections: JsonPayloadSections

    @property
    def needs_source_snapshot(self) -> bool:
        return self.sections.source_index or self.sections.finding_recipe_plan


class JsonPayloadProfile(Enum):
    """Named JSON payload profiles for CLI and programmatic callers."""

    full = JsonPayloadSections()
    agent = JsonPayloadSections(
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_structural_overlap=False,
        execution_plan_payload_mode=JsonExecutionPlanPayloadMode.COUNT_ONLY,
    )
    summary = JsonPayloadSections(
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        semantic_descent_graph=False,
        semantic_refactor_gate=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_structural_overlap=False,
        execution_plan_payload_mode=JsonExecutionPlanPayloadMode.COUNT_ONLY,
    )
    loop = JsonPayloadSections(
        finding_payload_mode=JsonFindingPayloadMode.counts_only,
        source_index=False,
        observation_graph=False,
        observation_fibers=False,
        semantic_descent_graph=False,
        semantic_refactor_gate=False,
        finding_recipe_plan=False,
        payload_timing=True,
        default_structural_overlap=False,
        execution_plan_payload_mode=JsonExecutionPlanPayloadMode.COUNT_ONLY,
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
class FocusedLoopColdAnalysisPolicy:
    """Choose a bounded, explicitly partial analysis for cold edit-loop scans."""

    json_enabled: bool
    payload_profile: JsonPayloadProfile
    has_report_filter: bool
    auto_context_enabled: bool
    explicit_context_roots: bool
    requires_full_analysis: bool

    @property
    def enabled(self) -> bool:
        return (
            self.json_enabled
            and self.payload_profile is JsonPayloadProfile.loop
            and self.has_report_filter
            and self.auto_context_enabled
            and not self.explicit_context_roots
            and not self.requires_full_analysis
        )


@dataclass(frozen=True)
class JsonPayloadStructuralOverlapPolicy:
    """Resolved structural-overlap policy for one CLI JSON payload profile."""

    explicit_request: bool | None
    json_enabled: bool
    payload_profile: JsonPayloadProfile

    @property
    def include_structural_overlap(self) -> bool:
        if self.explicit_request is not None:
            return self.explicit_request
        if self.json_enabled:
            return self.payload_profile.sections.default_structural_overlap
        return False


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
    structural_overlap_enabled: bool
    parsed_modules_required: bool
    analysis_cache_dir: Path | None
    focused_report_filter: bool = False

    @property
    def cache_lookup_enabled(self) -> bool:
        return (
            self.json_enabled
            and not self.structural_overlap_enabled
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
    cache_context: SemanticDescentGraphCacheContext

    def context(self) -> FastPreparseSemanticDescentContext:
        latest_graph = self.latest_graph()
        if latest_graph is None:
            return FastPreparseSemanticDescentContext(self.base_source)
        return FastPreparseSemanticDescentContext(
            SemanticDescentGraphAnalysisSource(
                cached_graph=latest_graph,
                cache_context=self.cache_context,
            ),
            latest_graph=latest_graph,
        )

    def latest_graph(self) -> SemanticDescentGraph | None:
        # Loop-summary lookup needs the graph only when an actual evidence-local
        # partial result is computed; ``FastCachedPathAnalysisAuthority`` loads
        # it lazily in that branch.  Eagerly unpickling the repository graph here
        # made every ordinary cache miss pay the full graph-load cost before AST
        # parsing, even when no previous analysis result could be reused.
        if (
            not self.preparse_cache_policy.uses_evidence_local_partial_reuse
            or not self.preparse_cache_policy.mode.requires_semantic_descent_cache
        ):
            return None
        return self.cache_context.latest_graph()


@dataclass(frozen=True)
class SourceSnapshotCacheEligibility:
    """Decide whether source-snapshot demand can use cached source context."""

    include_structural_overlap: bool
    codemod_plan_sequence: CodemodPlanSequence
    codemod_command_type: type[CliCommand] | None

    @property
    def needs_source_snapshot(self) -> bool:
        return (
            self.include_structural_overlap
            or self.codemod_plan_sequence.requires_source_snapshot
            or (
                self.codemod_command_type is not None
                and self.codemod_command_type.requires_source_snapshot()
            )
        )

    @property
    def requires_parsed_modules(self) -> bool:
        return self.codemod_plan_sequence.requires_source_snapshot or (
            self.codemod_command_type is not None
            and self.codemod_command_type.requires_parsed_modules()
        )

    @property
    def can_use_cached_source_context(self) -> bool:
        return self.needs_source_snapshot and not self.requires_parsed_modules


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

    summary: FindingSummary
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
class JsonScanStatus:
    """Machine-readable completeness contract for a scan result."""

    complete: bool
    mode: str
    analyzed_detector_count: int
    omitted_detector_count: int
    reason: str

    def to_dict(self) -> JsonObject:
        return {
            "complete": self.complete,
            "mode": self.mode,
            "analyzed_detector_count": self.analyzed_detector_count,
            "omitted_detector_count": self.omitted_detector_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JsonLoopCachePayloadBuilder:
    """Build loop JSON directly from an exact cache-summary hit."""

    summary: FindingSummary
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
class JsonSemanticDescentPayloadSource:
    """Repository graph source for the semantic-descent JSON section."""

    modules: list[ParsedModule]
    graph_source: SemanticDescentGraphAnalysisSource = field(
        default_factory=SemanticDescentGraphAnalysisSource
    )
    cached_repository_graph: SemanticDescentGraph | None = None

    @property
    def available(self) -> bool:
        return bool(self.modules) or self.cached_repository_graph is not None

    def repository_graph(self) -> SemanticDescentGraph:
        if self.cached_repository_graph is not None:
            return self.cached_repository_graph
        return self.graph_source.graph_for_modules(self.modules)


@dataclass(frozen=True)
class JsonPayloadBuilder:
    """Build the JSON report payload for one advisor scan."""

    findings: list[RefactorFinding]
    plans: list[RefactorPlan]
    modules: list[ParsedModule]
    economics: RefactorEvidenceEconomics | None = None
    change_budget: RepositoryChangeBudget | None = None
    timing: ScanTiming | None = None
    structural_overlap: StructuralOverlapReport | None = None
    execution_plan: RefactorExecutionPlanReport | None = None
    scan_guard_report: ArchitectureGuardReport | None = None
    source_snapshot: CodemodSourceSnapshot | None = None
    semantic_descent_source: JsonSemanticDescentPayloadSource | None = None
    payload_sections: JsonPayloadSections = JsonPayloadProfile.full.sections
    raw_findings: bool = False
    scan_status: JsonScanStatus | None = None

    def to_dict(self) -> JsonObject:
        payload_started = perf_counter()
        sections = self.payload_sections
        finding_tuple = tuple(self.findings)
        payload = JsonFindingPayloadEnvelope(
            summary=FindingSummary.from_findings(finding_tuple),
            section_policy=sections,
            finding_payload=JsonFindingPayloadProjection.payload(
                self.findings,
                sections.finding_payload_mode,
            ),
            plan_payload=tuple(plan.to_dict() for plan in self.plans),
        ).to_dict()
        if self.scan_status is not None:
            payload["scan_status"] = self.scan_status.to_dict()
        snapshot_demand = JsonPayloadSourceSnapshotDemand(sections=sections)
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
            payload["execution_plan"] = sections.execution_plan_payload_mode.payload(
                self.execution_plan
            )
        if self.structural_overlap is not None:
            payload["structural_overlap"] = self.structural_overlap.to_dict()
        semantic_refactor_gate_seconds = 0.0
        semantic_gate_report = SemanticRefactorGateReport.inactive()
        if sections.semantic_refactor_gate:
            started = perf_counter()
            semantic_gate_report = SemanticRefactorGateReport.from_findings(
                tuple(self.findings)
            )
            payload["semantic_refactor_gate"] = semantic_gate_report.to_dict()
            if semantic_gate_report.active:
                payload["findings"] = semantic_gate_report.finding_payload()
                payload["finding_payload_mode"] = (
                    JsonFindingPayloadMode.semantic_boundary_evidence.value
                )
                payload["active_finding_surface"] = (
                    "semantic_refactor_boundary_evidence"
                )
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


def load_codemod_plan_document(path: Path) -> CodemodPlanDocument:
    """Load caller-supplied codemod rewrites and guard invariants from JSON."""

    payload = cast(JsonObject, JsonDocumentSource(path).load())
    return CodemodPlanDocument.from_json_value(payload)


def load_codemod_plan_sequence(path: Path) -> CodemodPlanSequence:
    """Load one codemod document or staged codemod sequence from JSON."""

    return CodemodPlanRoot.from_json_value(
        JsonDocumentSource(path).load()
    ).as_sequence()


def load_codemod_plan_validation_payload(path: Path) -> JsonObject:
    """Load a codemod document or sequence and return its normalized JSON shape."""

    return CodemodPlanRoot.from_json_value(JsonDocumentSource(path).load()).to_dict()


def load_codemod_target_selector(path: Path) -> CodemodTargetSelector:
    """Load one registry-backed codemod target selector from JSON."""

    return CodemodTargetSelector.from_json_value(JsonDocumentSource(path).load())


def write_cli_json_artifact(path: Path | None, payload: JsonObject) -> None:
    """Write a machine-readable CLI artifact when the caller requested one."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


@dataclass(frozen=True)
class CodemodContinuationPlanArtifact:
    """Optional continuation-plan destination owned by a finding projection."""

    path: Path | None = None

    @property
    def requested(self) -> bool:
        return self.path is not None

    def write(self, report: CodemodProjectedFindingReport) -> None:
        write_cli_json_artifact(
            self.path,
            report.continuation_report.continuation_sequence.to_dict(),
        )


@dataclass(frozen=True)
class CodemodFindingProjectionAuthority:
    """Own the inputs and artifacts for rescanning one simulated source state."""

    config: DetectorConfig
    scope: AnalysisPathScope
    analysis_workers: int
    semantic_descent_source: SemanticDescentGraphAnalysisSource
    include_source_index: bool = False
    continuation_artifact: CodemodContinuationPlanArtifact = field(
        default_factory=CodemodContinuationPlanArtifact
    )

    def project(
        self,
        workflow_scan: CodemodWorkflowScan,
        simulation: CodemodSimulationReport,
        *,
        source_sequence: CodemodPlanSequence | None = None,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> CodemodProjectedFindingReport:
        report = CodemodSimulationFindingProjection(
            modules=tuple(workflow_scan.modules),
            findings=tuple(workflow_scan.findings),
            simulation=simulation,
            config=self.config,
            roots=self.scope.analysis_roots,
            report_roots=self.scope.report_roots,
            analysis_workers=self.analysis_workers,
            semantic_descent_source=self.semantic_descent_source,
            source_sequence=source_sequence,
            expected_removed_finding_ids=expected_removed_finding_ids,
            include_source_index=self.include_source_index,
            include_continuation=self.continuation_artifact.requested,
        ).report()
        self.continuation_artifact.write(report)
        return report


def format_codemod_refactor_goal_markdown(
    report: CodemodRefactorGoalReport,
) -> str:
    """Render a concise goal-directed codemod workflow summary."""

    return report.to_markdown()


def codemod_refactor_concept_from_args(
    args: argparse.Namespace,
) -> type[RefactorConcept]:
    """Resolve the nominal semantic migration requested at the CLI boundary."""

    try:
        concept_type = RefactorConcept.declaration_for_key(args.codemod_refactor_goal)
    except ValueError as error:
        choices = ", ".join(
            declaration.concept_key()
            for declaration in RefactorConcept.declaration_types()
        )
        raise ValueError(
            f"unknown codemod refactor goal {args.codemod_refactor_goal!r}; "
            f"choose one of {choices}"
        ) from error
    return concept_type


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
        return "No subsystem structural hypotheses."
    lines = ["Subsystem structural hypotheses (non-actionable):"]
    for index, plan in enumerate(plans, start=1):
        observed_patterns = ", ".join(
            (
                f"Pattern {pattern_id.value}: {pattern_id.display_name}"
                for pattern_id in plan.pattern_evidence.pattern_ids
            )
        )
        lines.append(f"{index}. {plan.subsystem}")
        lines.append(f"   - Summary: {plan.summary}")
        lines.append(f"   - Observed patterns: {observed_patterns}")
        lines.append(f"   - Certification: {plan.certification}")
        lines.append(f"   - Partial view: {plan.current_partial_view}")
        lines.append(
            f"   - Collapsed distinctions: {', '.join(plan.collapsed_distinctions)}"
        )
        lines.append(
            f"   - Missing capabilities: {', '.join(plan.missing_capabilities)}"
        )
        if plan.outcome.description_length_before:
            lines.append(
                "   - Semantic description length: "
                f"{plan.outcome.description_length_before} -> "
                f"{plan.outcome.description_length_after}; certified savings "
                f"{plan.outcome.description_length_savings}"
            )
        for title in plan.supporting_findings[:5]:
            lines.append(f"   - Supporting finding: {title}")
        for item in plan.evidence:
            lines.append(f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`")
    return "\n".join(lines)


def format_execution_plan_markdown(
    execution_plan: RefactorExecutionPlanReport,
) -> str:
    if not execution_plan.classes:
        return "No graph evidence classes."
    lines = [
        "Graph evidence classes (structural evidence only):",
        (
            "   - Summary: "
            f"{execution_plan.total_finding_count} finding(s), "
            f"{execution_plan.connected_component_count} connected component(s)"
        ),
    ]
    for index, execution_class in enumerate(execution_plan.classes, start=1):
        observed_patterns = ", ".join(
            (
                f"Pattern {pattern_id.value}: {pattern_id.display_name}"
                for pattern_id in execution_class.pattern_evidence.pattern_ids
            )
        )
        lines.append(f"{index}. {execution_class.subsystem}")
        lines.append(f"   - Class id: {execution_class.class_id}")
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
        lines.append(f"   - Observed patterns: {observed_patterns}")
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
    return f"{name} +{budget.added}/-{budget.deleted} (net {budget.net_added:+d})"


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
    economics: RefactorEvidenceEconomics,
    change_budget: RepositoryChangeBudget | None = None,
) -> str:
    lines = ["Evidence economics:"]
    lines.append(
        "   - Observed backend LOC savings: "
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
        "   - Evidence guard: "
        f"{'pass' if economics.evidence_guard_passes else 'fail'}; "
        f"{economics.proven_finding_count}/{economics.finding_count} findings "
        "carry LOC or semantic proof"
    )
    if economics.unproved_detector_ids:
        lines.append(
            "   - Detectors without payoff proof: "
            f"{', '.join(economics.unproved_detector_ids)}"
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


def format_structural_overlap_markdown(
    structural_overlap: StructuralOverlapReport,
) -> str:
    lines = [
        "Structural-overlap evidence (non-actionable):",
        "   - Observed keys: "
        f"{structural_overlap.observed_key_count}; groups: "
        f"{structural_overlap.group_count}",
        (
            "   - These groups share a current-snapshot coordinate. They do not "
            "prove an authority choice, finding removal, or trajectory."
        ),
    ]
    for group in structural_overlap.groups[:10]:
        lines.append(
            f"   - {group.key.axis.value} `{group.key.label}` -> "
            f"{group.finding_count} finding(s), "
            f"{group.detector_count} detector(s), "
            f"{group.file_count} file(s)"
        )
        lines.append("     detectors: " + ", ".join(group.detector_ids))
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
                "   - Use the gate to inspect authority proof obligations before "
                "evaluating any transformation."
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
        economics: RefactorEvidenceEconomics | None = None,
        change_budget: RepositoryChangeBudget | None = None,
        timing: ScanTiming | None = None,
        structural_overlap: StructuralOverlapReport | None = None,
        architecture_guard_report: ArchitectureGuardReport | None = None,
        raw_findings: bool = False,
    ) -> str:
        sections: list[str] = []
        semantic_gate_report = SemanticRefactorGateReport.from_findings(tuple(findings))
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
        if structural_overlap is not None:
            sections.append(format_structural_overlap_markdown(structural_overlap))
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
            pattern = finding.pattern_id
            lines.append(f"{index}. {finding.title}")
            lines.append(f"   - Stable id: {finding.stable_id}")
            lines.append(f"   - Pattern {pattern.value}: {pattern.display_name}")
            lines.append(f"   - Summary: {finding.summary}")
            lines.append(f"   - Capability gap: {finding.capability_gap}")
            lines.append(f"   - Required relation: {pattern.required_relation}")
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
            f"{scan.test_only_finding_count} test-only; "
            f"{scan.elapsed_seconds:.3f}s/{scan.scan_budget_seconds:.3f}s",
            f"     proof: {'pass' if scan.proof_passes else 'fail'}; "
            f"evidence guard: {'pass' if scan.economics.evidence_guard_passes else 'fail'}",
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
    selection_error_message: ClassVar[str] = "CLI commands are mutually exclusive"

    @classmethod
    def selected_type(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> type[Self] | None:
        selected_types = tuple(
            cast(type[Self], command_type)
            for command_type in CliCommand.__registry__.values()
            if issubclass(command_type, cls) and command_type.requested(args)
        )
        if len(selected_types) > 1:
            parser.error(cls.selection_error_message)
        return selected_types[0] if selected_types else None

    @classmethod
    @abstractmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        raise NotImplementedError

    @classmethod
    def require_execution_mode(
        cls,
        parser: argparse.ArgumentParser,
        mode: CodemodExecutionMode,
    ) -> None:
        if not mode.accepted_by(cls):
            parser.error("selected CLI command does not accept codemod execution modes")

    @classmethod
    def require_plan_input(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        if args.codemod_plan is not None and not issubclass(
            cls, CodemodPlanConsumingCliCommand
        ):
            parser.error("selected CLI command does not consume --codemod-plan")

    @classmethod
    def run_before_scan(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> int | None:
        return None

    @classmethod
    def requires_analysis(cls) -> bool:
        return True

    @classmethod
    def requires_parsed_modules(cls) -> bool:
        return False

    @classmethod
    def requires_source_snapshot(cls) -> bool:
        return False

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class CliEarlyExitCommand(CliCommand, ABC):
    """Registered command that can satisfy CLI execution before source scanning."""

    @classmethod
    def run_before_scan(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> int | None:
        return cls(parser, args).run()


class CodemodPlanProducingCliCommand(ABC):
    """Nominal family of CLI commands that emit reusable codemod plans."""


class CodemodPlanConsumingCliCommand(ABC):
    """Nominal family of CLI commands that consume supplied codemod plans."""


class CodemodExecutionModeCliCommand(ABC):
    """Nominal family accepting all codemod execution modes."""


class CodemodApplyExecutionModeCliCommand(ABC):
    """Nominal family accepting default and apply execution modes."""


@dataclass(frozen=True)
class CodemodScanCliCommand(CliCommand, ABC):
    """Registered codemod command executed against one prepared source scan."""

    source_snapshot: CodemodSourceSnapshot | None
    findings: list[RefactorFinding]
    modules: list[ParsedModule]
    config: DetectorConfig
    roots: tuple[Path, ...]
    report_roots: tuple[Path, ...]
    parse_cache_dir: Path | None
    semantic_descent_source: SemanticDescentGraphAnalysisSource
    execution_request: "CodemodPlanExecutionRequest"


class CodemodValidatePlanCliCommand(
    CliEarlyExitCommand,
    CodemodPlanProducingCliCommand,
    CodemodPlanConsumingCliCommand,
):
    """Validate a supplied codemod DSL plan and emit its normalized form."""

    command_id = "codemod_validate_plan"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_validate_plan

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


class CodemodComposePlansCliCommand(
    CliEarlyExitCommand,
    CodemodPlanProducingCliCommand,
):
    """Compose normalized codemod DSL plan documents."""

    command_id = "codemod_compose_plans"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_compose_plans is not None

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


class CodemodComposeSequenceCliCommand(
    CliEarlyExitCommand,
    CodemodPlanProducingCliCommand,
):
    """Compose normalized codemod DSL plans as ordered replay stages."""

    command_id = "codemod_compose_sequence"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_compose_sequence is not None

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
            file_path = parsed_module.file_path
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


class CodemodSourceSnapshotRequired(ABC):
    """Behavior shared by execution surfaces that require a source snapshot."""

    parser: argparse.ArgumentParser
    source_snapshot: CodemodSourceSnapshot | None
    source_snapshot_error_message: ClassVar[str] = (
        "codemod command requires a source snapshot"
    )

    def required_source_snapshot(self) -> CodemodSourceSnapshot:
        if self.source_snapshot is not None:
            return self.source_snapshot
        self.parser.error(self.source_snapshot_error_message)
        raise RuntimeError("argparse.error should have exited")


class CodemodPlanExecutionPresenter(ABC):
    """Nominal presentation boundary for typed plan-execution reports."""

    @abstractmethod
    def present_preflight(
        self,
        report: CodemodPlanPreflightReport,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def present_simulation(
        self,
        simulation: CodemodPlanSequenceSimulation,
        *,
        applied: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def present_operation_preflight_failure(
        self,
        report: CodemodOperationPreflightReport,
    ) -> None:
        raise NotImplementedError


class CodemodExecutionStrategy(ABC):
    """Leaf-owned execution semantics carried by a codemod mode member."""

    requested: ClassVar[bool] = True
    applies_changes: ClassVar[bool] = False
    unified_diff_requested: ClassVar[bool] = False
    requires_json_report: ClassVar[bool] = False
    allows_projected_findings: ClassVar[bool] = False

    @classmethod
    def accepted_by(cls, command_type: type[CliCommand]) -> bool:
        return issubclass(command_type, CodemodExecutionModeCliCommand)

    @classmethod
    def require_valid(
        cls,
        parser: argparse.ArgumentParser,
        *,
        projection_requested: bool,
    ) -> None:
        if projection_requested and not cls.allows_projected_findings:
            parser.error("--codemod-project-findings requires --codemod-simulate")

    @classmethod
    @abstractmethod
    def execute(
        cls,
        sequence: CodemodPlanSequence,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> int | None:
        raise NotImplementedError


class NoCodemodExecutionStrategy(CodemodExecutionStrategy):
    """No execution was requested."""

    requested = False

    @classmethod
    def accepted_by(cls, command_type: type[CliCommand]) -> bool:
        del command_type
        return True

    @classmethod
    def execute(
        cls,
        sequence: CodemodPlanSequence,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> None:
        del sequence, snapshot, presenter
        return None


class PreflightCodemodExecutionStrategy(CodemodExecutionStrategy):
    """Preflight one plan sequence without constructing rewritten source."""

    requires_json_report = True

    @classmethod
    def execute(
        cls,
        sequence: CodemodPlanSequence,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> int:
        report = sequence.preflight_snapshot(snapshot)
        presenter.present_preflight(report)
        return 0 if report.is_clean else 1


class SimulateCodemodExecutionStrategy(CodemodExecutionStrategy):
    """Simulate a plan and retain its unified diff without applying it."""

    unified_diff_requested = True
    requires_json_report = True
    allows_projected_findings = True

    @classmethod
    def execute(
        cls,
        sequence: CodemodPlanSequence,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> int:
        try:
            simulation = sequence.simulate_snapshot(snapshot)
        except CodemodOperationPreflightError as error:
            presenter.present_operation_preflight_failure(error.report)
            return 1
        applied = cls.applies_changes and simulation.is_clean
        if applied:
            simulation.apply()
        presenter.present_simulation(simulation, applied=applied)
        return 0 if simulation.is_clean else 1


class ApplyCodemodExecutionStrategy(SimulateCodemodExecutionStrategy):
    """Apply a clean plan simulation through its source-revision guards."""

    applies_changes = True
    unified_diff_requested = False
    requires_json_report = False
    allows_projected_findings = False

    @classmethod
    def accepted_by(cls, command_type: type[CliCommand]) -> bool:
        return super().accepted_by(command_type) or issubclass(
            command_type,
            CodemodApplyExecutionModeCliCommand,
        )


@dataclass(frozen=True)
class CodemodCliExecution(
    CodemodSourceSnapshotRequired,
    CodemodPlanExecutionPresenter,
):
    """Run the CLI codemod simulation/apply phase through plan-level DSL APIs."""

    source_snapshot_error_message: ClassVar[str] = (
        "--codemod-preflight/--codemod-simulate/--codemod-apply require a codemod plan"
    )
    parser: argparse.ArgumentParser
    args: argparse.Namespace
    source_snapshot: CodemodSourceSnapshot | None
    execution_request: "CodemodPlanExecutionRequest"
    workflow_scan: CodemodWorkflowScan | None = None

    def run(self) -> int | None:
        if not self.execution_request.mode.requested:
            return None
        snapshot = self.required_source_snapshot()
        return self.execution_request.execute(snapshot, self)

    def present_preflight(
        self,
        report: CodemodPlanPreflightReport,
    ) -> None:
        self.emit_preflight_report(report)

    def present_simulation(
        self,
        sequence_simulation: CodemodPlanSequenceSimulation,
        *,
        applied: bool,
    ) -> None:
        simulation = sequence_simulation.simulation
        snapshot = self.required_source_snapshot()
        architecture_guard_report = (
            sequence_simulation.architecture_guard_report
            if self.execution_request.sequence.has_architecture_guards
            else None
        )
        if not sequence_simulation.is_clean:
            self.emit_guard_failure(
                snapshot,
                simulation,
                architecture_guard_report,
            )
        else:
            self.emit_success(
                snapshot,
                simulation,
                applied,
                architecture_guard_report,
                sequence_simulation,
            )

    def present_operation_preflight_failure(
        self,
        report: CodemodOperationPreflightReport,
    ) -> None:
        self.emit_preflight_failure(report)

    def emit_preflight_failure(
        self,
        report: CodemodOperationPreflightReport,
    ) -> None:
        if self.execution_request.json_report_requested(self.args.json):
            print(
                json.dumps(
                    CodemodPreflightFailurePayload(report).to_dict(),
                    indent=2,
                )
            )
        else:
            print(f"Codemod preflight failed: {report.message}", file=sys.stderr)

    def emit_preflight_report(
        self,
        report: CodemodPlanPreflightReport,
    ) -> None:
        print(
            json.dumps(
                CodemodPlanPreflightPayload(report).to_dict(),
                indent=2,
            )
        )

    def emit_guard_failure(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        architecture_guard_report: ArchitectureGuardReport | None,
    ) -> None:
        if architecture_guard_report is None:
            raise RuntimeError("dirty codemod simulation requires architecture guards")
        if self.execution_request.json_report_requested(self.args.json):
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
            print(format_architecture_guard_markdown(architecture_guard_report))

    def emit_success(
        self,
        snapshot: CodemodSourceSnapshot,
        simulation: CodemodSimulationReport,
        applied: bool,
        architecture_guard_report: ArchitectureGuardReport | None,
        plan_sequence_simulation: CodemodPlanSequenceSimulation,
    ) -> None:
        if self.execution_request.json_report_requested(self.args.json):
            payload = CodemodSimulationPayload(
                simulation,
                applied=applied,
                post_guard_report=architecture_guard_report,
                unified_diff=self.optional_unified_diff(snapshot, simulation),
            ).to_dict()
            payload["plan_sequence_simulation"] = plan_sequence_simulation.to_dict()
            projected_findings = self.execution_request.projected_finding_report(
                self.workflow_scan,
                simulation,
                source_sequence=plan_sequence_simulation.sequence,
            )
            if projected_findings is not None:
                payload["projected_findings"] = projected_findings.to_dict()
            print(
                json.dumps(
                    payload,
                    indent=2,
                )
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


class CodemodExecutionMode(Enum):
    """Single authority for codemod execution semantics."""

    NONE = NoCodemodExecutionStrategy
    PREFLIGHT = PreflightCodemodExecutionStrategy
    SIMULATE = SimulateCodemodExecutionStrategy
    APPLY = ApplyCodemodExecutionStrategy

    @property
    def strategy(self) -> type[CodemodExecutionStrategy]:
        return self.value

    def accepted_by(self, command_type: type[CliCommand]) -> bool:
        """Return whether a nominal command family accepts this mode."""

        return self.strategy.accepted_by(command_type)

    @classmethod
    def from_namespace(
        cls,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> "CodemodExecutionMode":
        selected = tuple(
            mode
            for mode, supplied in (
                (cls.PREFLIGHT, args.codemod_preflight),
                (cls.SIMULATE, args.codemod_simulate),
                (cls.APPLY, args.codemod_apply),
            )
            if supplied
        )
        if len(selected) > 1:
            parser.error(
                "--codemod-preflight, --codemod-simulate, and --codemod-apply "
                "are mutually exclusive"
            )
        return selected[0] if selected else cls.NONE

    @property
    def requested(self) -> bool:
        return self.strategy.requested

    @property
    def applies_changes(self) -> bool:
        return self.strategy.applies_changes

    @property
    def unified_diff_requested(self) -> bool:
        return self.strategy.unified_diff_requested

    @property
    def requires_json_report(self) -> bool:
        return self.strategy.requires_json_report

    def require_valid(
        self,
        parser: argparse.ArgumentParser,
        *,
        projection_requested: bool,
    ) -> None:
        self.strategy.require_valid(
            parser,
            projection_requested=projection_requested,
        )

    def execute(
        self,
        sequence: CodemodPlanSequence,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> int | None:
        return self.strategy.execute(sequence, snapshot, presenter)


@dataclass(frozen=True)
class CodemodPlanExecutionRequest:
    """Codemod plan plus execution mode consumed by execution authorities."""

    sequence: CodemodPlanSequence
    mode: CodemodExecutionMode
    finding_projection: CodemodFindingProjectionAuthority | None = None

    def execute(
        self,
        snapshot: CodemodSourceSnapshot,
        presenter: CodemodPlanExecutionPresenter,
    ) -> int | None:
        return self.mode.execute(self.sequence, snapshot, presenter)

    def for_sequence(
        self,
        sequence: CodemodPlanSequence,
    ) -> "CodemodPlanExecutionRequest":
        return replace(self, sequence=sequence)

    def json_report_requested(self, json_flag: bool) -> bool:
        return (
            json_flag
            or self.mode.requires_json_report
            or self.finding_projection is not None
        )

    def projected_finding_report(
        self,
        workflow_scan: CodemodWorkflowScan | None,
        simulation: CodemodSimulationReport,
        *,
        source_sequence: CodemodPlanSequence | None = None,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> CodemodProjectedFindingReport | None:
        if self.finding_projection is None:
            return None
        if workflow_scan is None:
            raise RuntimeError("projected findings require a completed workflow scan")
        return self.finding_projection.project(
            workflow_scan,
            simulation,
            source_sequence=source_sequence,
            expected_removed_finding_ids=expected_removed_finding_ids,
        )

    @property
    def exact_recipe_execution(self) -> bool:
        return (
            self.mode.requested
            and self.finding_projection is None
            and self.sequence.has_recipes
            and not self.sequence.has_architecture_guards
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
            candidate_set=SourcePathCandidateSet.from_paths(
                cls.candidate_paths_for(requested_path, roots, cwd)
            ),
        )

    def source_mapping_entry(self) -> tuple[str, str] | None:
        file_path = self.unique_existing_file_path()
        if file_path is None:
            return None
        return (
            self.requested_path,
            Path(file_path).read_text(encoding="utf-8"),
        )

    def unique_existing_file_path(self) -> str | None:
        paths_by_resolved_path: dict[Path, str] = {}
        for file_path in self.candidate_set.paths:
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
        if self.sequence.has_unresolved_source_dependencies:
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
    CodemodScanCliCommand,
    CodemodSourceSnapshotRequired,
    ABC,
):
    """Registered command that emits one scan-backed codemod DSL query."""

    source_snapshot_error_message: ClassVar[str] = (
        "codemod scan query requires a source snapshot"
    )

    @classmethod
    def requires_source_snapshot(cls) -> bool:
        return True


class CodemodSynthesisExecutionCliCommand(
    CodemodScanQueryCliCommand,
    CodemodPlanProducingCliCommand,
    CodemodExecutionModeCliCommand,
    ABC,
):
    """Shared execution surface for finding-backed synthesis commands."""

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        return self.synthesis_execution(snapshot).run()

    @abstractmethod
    def synthesis_execution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "CodemodSynthesisExecution":
        raise NotImplementedError


class CodemodSynthesizePlanCliCommand(CodemodSynthesisExecutionCliCommand):
    """Emit finding-backed executable codemod recipes."""

    command_id = "codemod_synthesize_plan"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_synthesize_plan

    def synthesis_execution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "FindingRecipePlanSynthesisExecution":
        plan = snapshot.plan_from_findings(
            self.findings,
            detector_ids=tuple(self.args.codemod_goal_detectors),
        )
        return FindingRecipePlanSynthesisExecution(
            snapshot=snapshot,
            execution_request=self.execution_request.for_sequence(
                CodemodPlanSequence.from_document(plan.document)
            ),
            plan_out=self.args.codemod_plan_out,
            workflow_scan=CodemodWorkflowScan(self.modules, self.findings),
            plan=plan,
        )


class CodemodSynthesizeClassPlanCliCommand(CodemodSynthesisExecutionCliCommand):
    """Emit graph-clustered finding-backed typed codemod plans."""

    command_id = "codemod_synthesize_class_plan"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_synthesize_class_plan

    def synthesis_execution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> "FindingRecipeClassPlanSynthesisExecution":
        report = codemod_class_plan_from_findings(
            self.findings,
            root=self.roots[0],
            selector_context=snapshot,
            detector_ids=tuple(self.args.codemod_goal_detectors),
        )
        return FindingRecipeClassPlanSynthesisExecution(
            snapshot=snapshot,
            execution_request=self.execution_request.for_sequence(
                CodemodPlanSequence.from_document(report.finding_plan.document)
            ),
            plan_out=self.args.codemod_plan_out,
            workflow_scan=CodemodWorkflowScan(self.modules, self.findings),
            report=report,
        )


@dataclass(frozen=True)
class CodemodSynthesisExecution(
    CodemodPlanExecutionPresenter,
    ABC,
):
    """Execute and present one synthesized plan through its nominal envelope."""

    snapshot: CodemodSourceSnapshot
    execution_request: CodemodPlanExecutionRequest
    plan_out: Path | None
    workflow_scan: CodemodWorkflowScan

    @property
    @abstractmethod
    def finding_plan(self) -> FindingRecipePlan:
        raise NotImplementedError

    @abstractmethod
    def unexecuted_payload(self) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def preflight_payload(
        self,
        report: CodemodPlanPreflightReport,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def simulation_result_payload(
        self,
        simulation: FindingRecipePlanSimulation,
    ) -> JsonObject:
        raise NotImplementedError

    def simulation_payload(
        self,
        simulation: FindingRecipePlanSimulation,
        *,
        applied: bool,
    ) -> JsonObject:
        return {
            **self.simulation_result_payload(simulation),
            "applied": applied,
            "unified_diff": self.snapshot.unified_diff(simulation.simulation),
        }

    def operation_preflight_failure_payload(
        self,
        report: CodemodOperationPreflightReport,
    ) -> JsonObject:
        return {
            **self.unexecuted_payload(),
            **CodemodPreflightFailurePayload(report).to_dict(),
        }

    def with_projected_findings(
        self,
        payload: JsonObject,
        report: CodemodProjectedFindingReport,
    ) -> JsonObject:
        return {
            **payload,
            "projected_findings": report.to_dict(),
        }

    def run(self) -> int:
        synthesis_report = self.finding_plan.report
        if not synthesis_report.application_blocked:
            write_cli_json_artifact(
                self.plan_out,
                self.finding_plan.document.to_dict(),
            )
        if (
            self.execution_request.mode.applies_changes or self.plan_out is not None
        ) and synthesis_report.application_blocked:
            print(
                json.dumps(
                    {
                        **self.unexecuted_payload(),
                        "application_blocked": True,
                        "application_block_reason": (
                            synthesis_report.application_block_reason
                        ),
                    },
                    indent=2,
                )
            )
            return 1
        if not self.execution_request.mode.requested:
            print(json.dumps(self.unexecuted_payload(), indent=2))
            return 0
        exit_code = self.execution_request.execute(self.snapshot, self)
        if exit_code is None:
            raise RuntimeError("requested synthesized execution returned no exit code")
        return exit_code

    def present_preflight(
        self,
        report: CodemodPlanPreflightReport,
    ) -> None:
        print(json.dumps(self.preflight_payload(report), indent=2))

    def present_simulation(
        self,
        sequence_simulation: CodemodPlanSequenceSimulation,
        *,
        applied: bool,
    ) -> None:
        simulation = FindingRecipePlanSimulation.from_sequence_simulation(
            self.finding_plan,
            sequence_simulation,
        )
        payload = self.simulation_payload(simulation, applied=applied)
        projected_findings = self.execution_request.projected_finding_report(
            self.workflow_scan,
            simulation.simulation,
            source_sequence=sequence_simulation.sequence,
            expected_removed_finding_ids=(
                self.finding_plan.expected_removed_finding_ids
            ),
        )
        if projected_findings is not None:
            payload = self.with_projected_findings(payload, projected_findings)
        print(json.dumps(payload, indent=2))

    def present_operation_preflight_failure(
        self,
        report: CodemodOperationPreflightReport,
    ) -> None:
        print(
            json.dumps(
                self.operation_preflight_failure_payload(report),
                indent=2,
            )
        )


@dataclass(frozen=True)
class FindingRecipePlanSynthesisExecution(CodemodSynthesisExecution):
    """Flat JSON envelope for one synthesized finding plan."""

    plan: FindingRecipePlan

    @property
    def finding_plan(self) -> FindingRecipePlan:
        return self.plan

    def unexecuted_payload(self) -> JsonObject:
        return self.plan.to_dict()

    def preflight_payload(
        self,
        report: CodemodPlanPreflightReport,
    ) -> JsonObject:
        return FindingRecipePlanPreflight(
            plan=self.plan,
            preflight_report=report,
        ).to_dict()

    def simulation_result_payload(
        self,
        simulation: FindingRecipePlanSimulation,
    ) -> JsonObject:
        return simulation.to_dict()


@dataclass(frozen=True)
class FindingRecipeClassPlanSynthesisExecution(CodemodSynthesisExecution):
    """Class-grouped JSON envelope for one synthesized finding plan."""

    report: FindingRecipeClassPlanReport

    @property
    def finding_plan(self) -> FindingRecipePlan:
        return self.report.finding_plan

    def unexecuted_payload(self) -> JsonObject:
        return self.report.to_dict()

    def preflight_payload(
        self,
        report: CodemodPlanPreflightReport,
    ) -> JsonObject:
        return {
            **self.report.to_dict(),
            "preflight_report": report.to_dict(),
            "is_clean": report.is_clean,
        }

    def simulation_result_payload(
        self,
        simulation: FindingRecipePlanSimulation,
    ) -> JsonObject:
        return {
            **self.report.to_dict(),
            "simulation_result": simulation.to_dict(),
        }

    def with_projected_findings(
        self,
        payload: JsonObject,
        report: CodemodProjectedFindingReport,
    ) -> JsonObject:
        return {
            **super().with_projected_findings(payload, report),
            "class_plan_projected_deltas": report.class_plan_delta_report(
                self.report
            ).to_dict(),
        }


class CodemodSourceIndexCliCommand(CodemodScanQueryCliCommand):
    """Emit source-index target rows for DSL authoring."""

    command_id = "codemod_source_index"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_source_index

    @classmethod
    def requires_analysis(cls) -> bool:
        return False

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

    report_factory: ClassVar[CodemodSelectorReportFactory | None] = None

    def run(self) -> int:
        snapshot = self.required_source_snapshot()
        try:
            selector = load_codemod_target_selector(self.selector_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.parser.error(str(error))
        payload = self.payload_for_selector(snapshot, selector)
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
        if self.report_factory is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a report factory or override "
                "payload_for_selector"
            )
        return self.report_factory(snapshot, selector).to_dict()


class CodemodResolveSelectorCliCommand(CodemodSelectorQueryCliCommand):
    """Resolve one registry-backed target selector against scanned source."""

    command_id = "codemod_resolve_selector"
    report_factory = staticmethod(CodemodSourceSnapshot.resolve_selector)

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_resolve_selector is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_resolve_selector


class CodemodTargetSourceCliCommand(CodemodSelectorQueryCliCommand):
    """Emit exact source spans for one resolved target selector."""

    command_id = "codemod_target_source"
    report_factory = staticmethod(CodemodSourceSnapshot.target_source_report)

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_target_source is not None

    @property
    def selector_path(self) -> Path:
        return self.args.codemod_target_source


class CodemodRefactorGoalCliCommand(
    CodemodScanCliCommand,
    CodemodPlanProducingCliCommand,
    CodemodPlanConsumingCliCommand,
    CodemodApplyExecutionModeCliCommand,
):
    """Execute one goal-directed staged refactor against a prepared scan."""

    command_id = "codemod_refactor_goal"

    @classmethod
    def requested(cls, args: argparse.Namespace) -> bool:
        return args.codemod_refactor_goal is not None

    @classmethod
    def requires_parsed_modules(cls) -> bool:
        return True

    def run(self) -> int:
        if self.execution_request.sequence.has_recipes:
            self.parser.error(
                "--codemod-refactor-goal accepts guard-only --codemod-plan input"
            )
        try:
            migration_type = codemod_refactor_concept_from_args(self.args)
        except ValueError as error:
            self.parser.error(str(error))
        report = CodemodRefactorGoalRunner(
            resolved_dir=self.parse_cache_dir,
            enabled=self.args.use_parse_cache,
            roots=self.roots,
            report_roots=self.report_roots,
            config=self.config,
            parse_workers=self.args.parse_workers,
            guard_suite=self.execution_request.sequence.guard_suite,
            dry_run=not self.execution_request.mode.applies_changes,
            initial_scan=CodemodWorkflowScan(
                modules=self.modules,
                findings=self.findings,
            ),
            migration_type=migration_type,
            trajectory_budget=CodemodRefactorTrajectoryBudget(
                max_depth=self.args.codemod_goal_max_stages,
                max_states=self.args.codemod_goal_max_states,
                recipe_frontier=FindingRecipeFrontierBudget(
                    max_candidate_batches=self.args.codemod_goal_max_branches,
                ),
            ),
        ).run()
        replay_plan_payload = report.replay_sequence.to_dict()
        if report.stop_reason.completed:
            write_cli_json_artifact(self.args.codemod_plan_out, replay_plan_payload)
        if self.args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(format_codemod_refactor_goal_markdown(report))
        return 0 if report.stop_reason.completed else 1


def _main_without_deadline() -> int:
    """Run the command-line interface and return a process status code."""
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for nominal architecture."
    )
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)
    args = parser.parse_args()

    selected_command_type = CliCommand.selected_type(parser, args)
    scan_analysis_required = (
        selected_command_type is None or selected_command_type.requires_analysis()
    )
    if args.codemod_plan_out is not None and not (
        selected_command_type is not None
        and issubclass(selected_command_type, CodemodPlanProducingCliCommand)
    ):
        parser.error("--codemod-plan-out requires a plan-producing codemod command")
    codemod_execution_mode = CodemodExecutionMode.from_namespace(args, parser)
    codemod_execution_mode.require_valid(
        parser,
        projection_requested=args.codemod_project_findings,
    )
    if selected_command_type is not None:
        selected_command_type.require_execution_mode(
            parser,
            codemod_execution_mode,
        )
        selected_command_type.require_plan_input(parser, args)
        early_exit_code = selected_command_type.run_before_scan(parser, args)
        if early_exit_code is not None:
            return early_exit_code

    config = DetectorConfig.from_namespace(args)
    try:
        json_payload_profile = JsonPayloadProfile.from_cli_value(args.json_payload)
    except ValueError as error:
        parser.error(str(error))
    structural_overlap_policy = JsonPayloadStructuralOverlapPolicy(
        explicit_request=args.include_structural_overlap,
        json_enabled=args.json,
        payload_profile=json_payload_profile,
    )
    args.include_structural_overlap = (
        structural_overlap_policy.include_structural_overlap
    )
    JsonDocumentInputSet.from_option_paths(
        (
            ("--codemod-plan", (args.codemod_plan,)),
            ("--codemod-resolve-selector", (args.codemod_resolve_selector,)),
            ("--codemod-target-source", (args.codemod_target_source,)),
        )
    ).require_at_most_one_stdin(parser)
    codemod_requested = (
        args.codemod_plan is not None
        or codemod_execution_mode.requested
        or selected_command_type is not None
    )
    if (
        args.codemod_continuation_plan_out is not None
        and not args.codemod_project_findings
    ):
        parser.error(
            "--codemod-continuation-plan-out requires --codemod-project-findings"
        )
    if args.codemod_project_source_index and not args.codemod_project_findings:
        parser.error(
            "--codemod-project-source-index requires --codemod-project-findings"
        )
    codemod_plan_sequence = (
        load_codemod_plan_sequence(args.codemod_plan)
        if args.codemod_plan is not None
        else CodemodPlanSequence()
    )
    if (
        codemod_execution_mode.requested
        and selected_command_type is None
        and not codemod_plan_sequence.requires_source_snapshot
    ):
        parser.error("codemod execution requires --codemod-plan")
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
    if args.use_parse_cache and args.cache_dir is None:
        maintain_default_cache(root)
    analysis_cache_dir = analysis_cache_dir_for_root(
        root,
        parse_cache_dir,
        args.use_parse_cache,
    )
    source_policy = PythonSourcePathPolicy(include_tests=args.include_tests)
    semantic_descent_cache_context = SemanticDescentGraphCacheContext.from_parse_cache(
        roots,
        parse_cache_dir,
        args.use_parse_cache,
        source_policy,
    )
    semantic_descent_analysis_source = SemanticDescentGraphAnalysisSource(
        cache_context=semantic_descent_cache_context,
    )
    finding_projection = (
        CodemodFindingProjectionAuthority(
            config=config,
            scope=path_scope,
            analysis_workers=args.analysis_workers,
            semantic_descent_source=semantic_descent_analysis_source,
            include_source_index=args.codemod_project_source_index,
            continuation_artifact=CodemodContinuationPlanArtifact(
                args.codemod_continuation_plan_out
            ),
        )
        if args.codemod_project_findings
        else None
    )
    codemod_execution_request = CodemodPlanExecutionRequest(
        sequence=codemod_plan_sequence,
        mode=codemod_execution_mode,
        finding_projection=finding_projection,
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

    fast_codemod_source_snapshot = None
    if (
        codemod_execution_request.exact_recipe_execution
        and not args.include_structural_overlap
        and args.import_lean_export is None
        and selected_command_type is None
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
            execution_request=codemod_execution_request,
        ).run()
        if fast_codemod_execution_result is not None:
            return fast_codemod_execution_result

    source_snapshot_cache_eligibility = SourceSnapshotCacheEligibility(
        include_structural_overlap=args.include_structural_overlap,
        codemod_plan_sequence=codemod_plan_sequence,
        codemod_command_type=selected_command_type,
    )
    cached_source_context = None
    if args.import_lean_export is None:
        preparse_cache_policy = JsonSummaryPreparseCachePolicy(
            json_enabled=args.json,
            payload_profile=json_payload_profile,
            structural_overlap_enabled=args.include_structural_overlap,
            parsed_modules_required=(
                source_snapshot_cache_eligibility.requires_parsed_modules
            ),
            analysis_cache_dir=analysis_cache_dir,
            focused_report_filter=path_scope.has_report_filter,
        )
        fast_cache_result = None
        cached_semantic_descent_graph = None
        analysis_cache_identity = None
        scan_status = None
        preparse_cache_mode = preparse_cache_policy.mode
        focused_loop_cold_policy = FocusedLoopColdAnalysisPolicy(
            json_enabled=args.json,
            payload_profile=json_payload_profile,
            has_report_filter=path_scope.has_report_filter,
            auto_context_enabled=args.auto_context_root,
            explicit_context_roots=bool(args.context_roots),
            requires_full_analysis=(
                args.include_structural_overlap
                or preparse_cache_policy.parsed_modules_required
                or args.include_execution_plan
                or args.include_plans
                or args.plans_only
                or codemod_requested
            ),
        )
        source_context_cache_lookup_enabled = (
            source_snapshot_cache_eligibility.can_use_cached_source_context
            and analysis_cache_dir is not None
            and args.use_parse_cache
            and not args.include_structural_overlap
        )
        cache_lookup_enabled = (
            preparse_cache_policy.cache_lookup_enabled and preparse_cache_mode.enabled
        ) or source_context_cache_lookup_enabled
        if scan_analysis_required and cache_lookup_enabled:
            started = perf_counter()
            fast_semantic_descent_context = FastPreparseSemanticDescentSourceAuthority(
                preparse_cache_policy=preparse_cache_policy,
                base_source=semantic_descent_analysis_source,
                cache_context=semantic_descent_cache_context,
            ).context()
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
                report_roots=path_scope.report_roots,
                reuse_policy=(
                    preparse_cache_mode.reuse_policy(
                        focused_report_filter=path_scope.has_report_filter
                    )
                    if preparse_cache_mode.enabled
                    else FastCacheReusePolicy.EXACT_ONLY
                ),
                semantic_descent_source=fast_semantic_descent_analysis_source,
            )
            fast_cache_authority = FastCachedPathAnalysisAuthority(fast_cache_request)
            if (
                json_payload_profile is JsonPayloadProfile.loop
                and not path_scope.has_report_filter
                and not args.include_execution_plan
                and not args.include_plans
                and not args.plans_only
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
                    semantic_descent_cache_context.cached_graph()
                )
                if (
                    cached_semantic_descent_graph is None
                    and preparse_cache_policy.uses_evidence_local_partial_reuse
                ):
                    cached_semantic_descent_graph = latest_semantic_descent_graph
                if cached_semantic_descent_graph is None:
                    fast_cache_result = None
            if fast_cache_result is not None and source_context_cache_lookup_enabled:
                source_context_lookup = CodemodSourceContextCache(
                    analysis_cache_dir
                ).load(fast_cache_result.cache_identity)
                if source_context_lookup.status is AnalysisCacheStatus.HIT:
                    cached_source_context = source_context_lookup.context
                else:
                    fast_cache_result = None
            fast_cache_seconds = round(perf_counter() - started, 3)
        if fast_cache_result is not None:
            modules = []
            parse_seconds = 0.0
            analysis_cache_status = fast_cache_result.cache_status
            analysis_cache_identity = fast_cache_result.cache_identity
            findings = path_scope.filter_findings(fast_cache_result.findings)
            analysis_seconds = fast_cache_seconds
            if analysis_cache_status is AnalysisCacheStatus.PARTIAL:
                detector_types = default_detector_types_for_analysis()
                analyzed_detector_count = len(
                    EvidenceLocalPartialDetectorSelection.from_detector_types(
                        detector_types
                    ).rerun_detector_family
                )
                scan_status = JsonScanStatus(
                    complete=False,
                    mode="focused_cache_partial",
                    analyzed_detector_count=analyzed_detector_count,
                    omitted_detector_count=(
                        len(detector_types) - analyzed_detector_count
                    ),
                    reason="changed_files_reuse_evidence_local_detectors",
                )
        elif focused_loop_cold_policy.enabled:
            detector_types = default_detector_types_for_analysis()
            detector_partition = DetectorTypePartition.from_detector_types(
                detector_types
            )
            local_detector_types = detector_partition.per_module_detector_types
            parse_elapsed = 0.0
            analysis_elapsed = 0.0
            findings = []
            module_cache_statuses: list[AnalysisCacheStatus] = []
            seen_report_paths: set[Path] = set()
            for report_root in path_scope.report_roots:
                normalized_report_path = report_root.resolve()
                if normalized_report_path in seen_report_paths:
                    continue
                seen_report_paths.add(normalized_report_path)
                started = perf_counter()
                local_modules = parse_python_module_roots(
                    (report_root,),
                    cache_dir=parse_cache_dir,
                    use_parse_cache=args.use_parse_cache,
                    parse_workers=args.parse_workers,
                    source_policy=source_policy,
                )
                parse_elapsed += perf_counter() - started
                started = perf_counter()
                for local_module in local_modules:
                    module_result = analyze_module_detector_types_with_cache(
                        local_module,
                        config,
                        detector_types=local_detector_types,
                        presentation_roots=roots,
                        analysis_cache_dir=analysis_cache_dir,
                    )
                    findings.extend(module_result.findings)
                    module_cache_statuses.append(module_result.cache_status)
                release_module_analysis_memory()
                analysis_elapsed += perf_counter() - started
                del local_modules
            modules = []
            findings = SortedFindingsAuthority.sort(findings)
            parse_seconds = round(parse_elapsed, 3)
            analysis_seconds = round(analysis_elapsed, 3)
            analysis_cache_status = (
                AnalysisCacheStatus.HIT
                if module_cache_statuses
                and all(
                    status is AnalysisCacheStatus.HIT
                    for status in module_cache_statuses
                )
                else (
                    AnalysisCacheStatus.PARTIAL
                    if AnalysisCacheStatus.HIT in module_cache_statuses
                    else (
                        AnalysisCacheStatus.DISABLED
                        if module_cache_statuses
                        and all(
                            status is AnalysisCacheStatus.DISABLED
                            for status in module_cache_statuses
                        )
                        else AnalysisCacheStatus.MISS
                    )
                )
            )
            scan_status = JsonScanStatus(
                complete=False,
                mode="focused_local_partial",
                analyzed_detector_count=len(local_detector_types),
                omitted_detector_count=(
                    len(detector_types) - len(local_detector_types)
                ),
                reason="cold_auto_context_omits_context_dependent_detectors",
            )
        elif (
            args.json
            and json_payload_profile.sections.compact_analysis_compatible
            and not preparse_cache_policy.parsed_modules_required
            and scan_analysis_required
            and not codemod_requested
            and not DetectorTypePartition.from_detector_types(
                default_detector_types_for_analysis()
            ).ast_retaining_context_detector_types
        ):
            requested_compact_semantic_graph = (
                json_payload_profile.sections.semantic_descent_graph
            )
            if requested_compact_semantic_graph:
                cached_semantic_descent_graph = (
                    semantic_descent_cache_context.cached_graph()
                )
            compact_result = analyze_compact_roots_with_cache(
                roots,
                config,
                cache_dir=parse_cache_dir,
                analysis_cache_dir=analysis_cache_dir,
                use_parse_cache=args.use_parse_cache,
                parse_workers=args.parse_workers,
                source_policy=source_policy,
                report_scope=path_scope,
                include_semantic_descent_graph=(
                    requested_compact_semantic_graph
                    and cached_semantic_descent_graph is None
                ),
            )
            modules = []
            findings = compact_result.findings
            built_semantic_descent_graph = compact_result.semantic_descent_graph
            if built_semantic_descent_graph is not None:
                cached_semantic_descent_graph = built_semantic_descent_graph
                semantic_descent_cache_context.store_exact_graph(
                    built_semantic_descent_graph
                )
            parse_seconds = round(compact_result.preparation_seconds, 3)
            analysis_seconds = round(compact_result.analysis_seconds, 3)
            analysis_cache_status = compact_result.cache_status
            analysis_cache_identity = compact_result.cache_identity
            detector_types = default_detector_types_for_analysis()
            scan_status = JsonScanStatus(
                complete=True,
                mode="exact_compact_global",
                analyzed_detector_count=len(detector_types),
                omitted_detector_count=0,
                reason="all_context_detectors_use_compact_global_projections",
            )
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
            if not scan_analysis_required:
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
                    report_scope=path_scope,
                )
                unfiltered_findings = analysis_result.findings
                analysis_cache_status = analysis_result.cache_status
                analysis_cache_identity = analysis_result.cache_identity
                findings = path_scope.filter_findings(unfiltered_findings)
                analysis_seconds = round(perf_counter() - started, 3)
    else:
        modules = []
        findings = analyze_lean_export(args.import_lean_export)
        parse_seconds = 0.0
        analysis_seconds = 0.0
        analysis_cache_status = None
        cached_semantic_descent_graph = None
        analysis_cache_identity = None
        scan_status = None
    plans = None
    execution_plan = None
    planning_seconds = 0.0
    source_index_seconds = 0.0
    if args.include_plans or args.plans_only or args.include_execution_plan:
        started = perf_counter()
        if args.include_plans or args.plans_only:
            plans = build_refactor_plans(findings, root)
        if args.include_execution_plan or args.plans_only:
            execution_plan_cache = AnalysisFindingCache(analysis_cache_dir)
            execution_plan_cache_identity = (
                AnalysisExecutionPlanCacheIdentity.from_analysis_identity(
                    analysis_cache_identity,
                    root,
                    path_scope.report_roots,
                )
                if analysis_cache_identity is not None
                else None
            )
            execution_plan_lookup = (
                execution_plan_cache.load_execution_plan(execution_plan_cache_identity)
                if execution_plan_cache_identity is not None
                else None
            )
            if (
                execution_plan_lookup is not None
                and execution_plan_lookup.status is AnalysisCacheStatus.HIT
            ):
                execution_plan = execution_plan_lookup.plan
            else:
                execution_plan = build_refactor_execution_plan(findings, root)
                if execution_plan_cache_identity is not None:
                    execution_plan_cache.store_execution_plan(
                        execution_plan_cache_identity,
                        execution_plan,
                    )
        planning_seconds = round(perf_counter() - started, 3)
    include_economics = args.include_economics or args.include_change_budget
    economics = (
        RefactorEvidenceEconomics.from_findings_and_plans(findings, plans or [])
        if include_economics
        else None
    )
    change_budget = (
        RepositoryChangeBudget.from_git_diff(root, compare_ref=args.compare_ref)
        if args.include_change_budget
        else None
    )
    architecture_guard_rules = codemod_plan_sequence.guard_suite.to_tuple()
    architecture_guard_evaluator = ArchitectureGuardSourceEvaluator(
        modules,
        architecture_guard_rules,
    )
    structural_overlap = None
    architecture_guard_report = None
    source_snapshot = None
    if source_snapshot_cache_eligibility.needs_source_snapshot:
        started = perf_counter()
        if (
            cached_source_context is not None
            and source_snapshot_cache_eligibility.can_use_cached_source_context
        ):
            source_snapshot = cached_source_context.snapshot_for_findings(
                findings,
                parse_workers=args.parse_workers,
            )
        elif source_snapshot_cache_eligibility.can_use_cached_source_context:
            source_context = CodemodSourceContext.from_modules(modules, findings)
            CodemodSourceContextCache(analysis_cache_dir).store(
                analysis_cache_identity,
                source_context,
            )
            source_snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
        else:
            source_snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
        source_index_seconds = round(perf_counter() - started, 3)

    if selected_command_type is not None:
        return cast(type[CodemodScanCliCommand], selected_command_type)(
            parser,
            args,
            source_snapshot,
            findings,
            modules,
            config,
            roots,
            path_scope.report_roots,
            parse_cache_dir,
            semantic_descent_analysis_source,
            codemod_execution_request,
        ).run()

    if args.include_structural_overlap:
        source_index = source_snapshot.source_index
        structural_overlap = build_structural_overlap_report(
            findings,
            source_index,
            limits=StructuralOverlapReportLimits(
                maximum_group_count=args.structural_overlap_max_groups,
                minimum_finding_count=args.structural_overlap_min_findings,
            ),
        )
    if architecture_guard_rules:
        architecture_guard_report = architecture_guard_evaluator.report_for_snapshot(
            source_snapshot
        )
    if (
        not args.include_structural_overlap
        and not codemod_plan_sequence.requires_source_snapshot
    ):
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
        execution_request=codemod_execution_request,
        workflow_scan=CodemodWorkflowScan(modules, findings),
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
                    structural_overlap=structural_overlap,
                    execution_plan=execution_plan,
                    scan_guard_report=architecture_guard_report,
                    source_snapshot=source_snapshot,
                    semantic_descent_source=JsonSemanticDescentPayloadSource(
                        modules=modules,
                        graph_source=semantic_descent_analysis_source,
                        cached_repository_graph=cached_semantic_descent_graph,
                    ),
                    payload_sections=json_payload_profile.sections,
                    raw_findings=args.raw_findings,
                    scan_status=scan_status,
                ).to_dict(),
                indent=2,
            )
        )
    else:
        if args.plans_only:
            sections = []
            semantic_gate_report = SemanticRefactorGateReport.from_findings(
                tuple(findings)
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
            if structural_overlap is not None:
                sections.append(format_structural_overlap_markdown(structural_overlap))
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
                    structural_overlap=structural_overlap,
                    architecture_guard_report=architecture_guard_report,
                    raw_findings=args.raw_findings,
                )
            )
    return 0


@dataclass(frozen=True)
class CliScanDeadlineRequest:
    """Project standard-scan deadline intent before full CLI dispatch."""

    budget_seconds: float
    json_enabled: bool

    @classmethod
    def from_argv(cls, argv: tuple[str, ...]) -> "CliScanDeadlineRequest | None":
        if any(
            option in argv
            for option in (
                "--help",
                "-h",
                "--prove-economics",
                "--predict-scan",
                "--calibrate",
            )
        ):
            return None
        budget_seconds = 20.0
        for index, argument in enumerate(argv):
            if argument.startswith("--scan-budget-seconds="):
                raw_budget = argument.split("=", 1)[1]
            elif argument == "--scan-budget-seconds" and index + 1 < len(argv):
                raw_budget = argv[index + 1]
            else:
                continue
            try:
                budget_seconds = float(raw_budget)
            except ValueError:
                return None
            break
        if budget_seconds <= 0.0:
            return None
        return cls(
            budget_seconds=budget_seconds,
            json_enabled="--json" in argv,
        )

    def timeout_payload(self, error: ScanDeadlineExceeded) -> JsonObject:
        return {
            "findings": [],
            "plans": [],
            "finding_payload_mode": "deadline_incomplete",
            "finding_count": None,
            "finding_counts": None,
            "scan_status": {
                "complete": False,
                "deadline_exceeded": True,
                "stage": error.stage,
                "budget_seconds": round(error.budget_seconds, 3),
                "elapsed_seconds": round(error.elapsed_seconds, 3),
            },
        }

    def emit_timeout(self, error: ScanDeadlineExceeded) -> None:
        if self.json_enabled:
            print(json.dumps(self.timeout_payload(error), indent=2))
            return
        print(str(error), file=sys.stderr)

    def terminate_process(self, error: ScanDeadlineExceeded) -> None:
        """Publish the incomplete result and terminate without unwinding workers."""

        for child in multiprocessing.active_children():
            child.terminate()
        self.emit_timeout(error)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(124)


def main(*, hard_exit_on_deadline: bool = False) -> int:
    """Run the CLI under the declared absolute scan wall-clock budget."""

    request = CliScanDeadlineRequest.from_argv(tuple(sys.argv[1:]))
    if request is None:
        return _main_without_deadline()
    deadline = ScanDeadline.start(request.budget_seconds)
    try:
        with enforce_scan_deadline(
            deadline,
            hard_timeout=(request.terminate_process if hard_exit_on_deadline else None),
        ):
            return _main_without_deadline()
    except ScanDeadlineExceeded as error:
        if hard_exit_on_deadline:
            request.terminate_process(error)
        request.emit_timeout(error)
        return 124


def process_main() -> int:
    """Run the installed command with prompt deadline termination semantics."""

    return main(hard_exit_on_deadline=True)


if __name__ == "__main__":
    raise SystemExit(process_main())
