"""CLI and top-level analysis helpers.

This module contains the programmatic entrypoints used by tests and automation as
well as the command-line interface used by developers. The public helpers are the
recommended way to analyze a path or synthesize subsystem plans from findings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .analysis import analyze_modules, analyze_path, plan_path
from .ast_tools import parse_python_modules
from .calibration import (
    CalibrationReport,
    format_calibration_markdown,
    run_calibration_manifest,
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
from .models import AnalysisReport, RefactorFinding, RefactorPlan
from .observation_graph import build_observation_graph
from .patterns import PATTERN_SPECS
from .planner import build_refactor_plans

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


@dataclass(frozen=True)
class CliArgumentSpec:
    flags: tuple[str, ...]
    help: str
    action: str | None = None
    default: object | None = None
    dest: str | None = None
    nargs: str | int | None = None
    value_type: type[object] | None = None

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        kwargs: dict[str, object] = {"help": self.help}
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
            flags=("path",),
            nargs="?",
            default="nominal_refactor_advisor",
            help="Root path to analyze (defaults to nominal_refactor_advisor).",
        ),
        CliArgumentSpec(
            flags=("--json",),
            action="store_true",
            help="Emit JSON instead of Markdown.",
        ),
        CliArgumentSpec(
            flags=("--include-plans",),
            action="store_true",
            help="Also synthesize subsystem-level composed refactor plans.",
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
            flags=("--prove-economics",),
            action="store_true",
            help="Run the standard long-term economics proof report.",
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


def _json_payload(
    findings: list[RefactorFinding],
    plans: list[RefactorPlan],
    modules: list,
    economics: RecommendationEconomics | None = None,
    change_budget: RepositoryChangeBudget | None = None,
) -> dict[str, object]:
    report = AnalysisReport(findings=tuple(findings), plans=tuple(plans))
    graph = build_observation_graph(modules)
    payload = report.to_dict()
    payload["observations"] = [asdict(item) for item in graph.observations]
    payload["fibers"] = [asdict(item) for item in graph.fibers]
    if economics is not None:
        payload["economics"] = economics.to_dict()
    if change_budget is not None:
        payload["change_budget"] = change_budget.to_dict()
    return payload


def _format_markdown(
    findings: list[RefactorFinding],
    plans: list[RefactorPlan] | None = None,
    economics: RecommendationEconomics | None = None,
    change_budget: RepositoryChangeBudget | None = None,
) -> str:
    sections: list[str] = []
    if findings:
        sections.append(_format_findings_markdown(findings))
    elif not plans:
        sections.append("No refactoring findings.")
    if plans is not None:
        sections.append(_format_plans_markdown(plans))
    if economics is not None:
        sections.append(_format_economics_markdown(economics, change_budget))
    return "\n\n".join(section for section in sections if section)


def _format_findings_markdown(findings: list[RefactorFinding]) -> str:
    if not findings:
        return "No refactoring findings."
    lines: list[str] = []
    for index, finding in enumerate(findings, start=1):
        pattern = PATTERN_SPECS[finding.pattern_id]
        lines.append(f"{index}. {finding.title}")
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
                f"certified savings {certificate.certified_description_length_savings}"
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
            lines.append(f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`")
    return "\n".join(lines)


def _format_plans_markdown(plans: list[RefactorPlan]) -> str:
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


def _format_change_budget_item(name: str, budget: LineChangeBudget) -> str:
    return f"{name} +{budget.added}/-{budget.deleted} " f"(net {budget.net_added:+d})"


def _format_economics_markdown(
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


def _format_scan_proof_markdown(scan: ScanEconomicsProof) -> list[str]:
    lines = [
        f"   - {scan.label}: {scan.finding_count} finding(s), "
        f"{scan.production_finding_count} production, "
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


def _format_economics_proof_markdown(report: EconomicsProofReport) -> str:
    lines = [
        "Economics proof:",
        f"   - Overall: {'pass' if report.proof_passes else 'fail'}",
    ]
    if report.regression_reasons:
        lines.append("   - Regression reasons: " + ", ".join(report.regression_reasons))
    lines.extend(_format_scan_proof_markdown(report.package_scan))
    lines.extend(_format_scan_proof_markdown(report.repository_scan))
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


def _proof_exit_code(
    report: EconomicsProofReport, *, fail_on_proof_regression: bool
) -> int:
    if fail_on_proof_regression and not report.proof_passes:
        return 1
    return 0


def _calibration_exit_code(
    report: CalibrationReport, *, fail_on_calibration_regression: bool
) -> int:
    if fail_on_calibration_regression and not report.passes:
        return 1
    return 0


def main() -> int:
    """Run the command-line interface and return a process status code."""
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for nominal architecture."
    )
    for spec in _CLI_ARGUMENT_SPECS:
        spec.add_to_parser(parser)
    args = parser.parse_args()

    config = DetectorConfig.from_namespace(args)
    if args.calibrate is not None:
        calibration_report = run_calibration_manifest(args.calibrate, config=config)
        if args.json:
            print(json.dumps(calibration_report.to_dict(), indent=2))
        else:
            print(format_calibration_markdown(calibration_report))
        return _calibration_exit_code(
            calibration_report,
            fail_on_calibration_regression=args.fail_on_calibration_regression,
        )

    root = Path(args.path)
    if args.prove_economics:
        proof_report = build_economics_proof_report(
            root,
            config=config,
            compare_ref=args.compare_ref,
            scan_budget_seconds=args.scan_budget_seconds,
        )
        if args.json:
            print(json.dumps(proof_report.to_dict(), indent=2))
        else:
            print(_format_economics_proof_markdown(proof_report))
        return _proof_exit_code(
            proof_report,
            fail_on_proof_regression=args.fail_on_proof_regression,
        )

    modules = parse_python_modules(root)
    findings = analyze_modules(modules, config)
    plans = None
    if args.include_plans or args.plans_only:
        plans = build_refactor_plans(findings, root)
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
    if args.json:
        json_findings = [] if args.plans_only else findings
        print(
            json.dumps(
                _json_payload(
                    json_findings,
                    plans or [],
                    modules,
                    economics=economics,
                    change_budget=change_budget,
                ),
                indent=2,
            )
        )
    else:
        if args.plans_only:
            sections = [_format_plans_markdown(plans or [])]
            if economics is not None:
                sections.append(_format_economics_markdown(economics, change_budget))
            print("\n\n".join(sections))
        else:
            print(
                _format_markdown(
                    findings,
                    plans,
                    economics=economics,
                    change_budget=change_budget,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
