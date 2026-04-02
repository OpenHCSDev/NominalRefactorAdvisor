from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .ast_tools import build_observation_graph, parse_python_modules
from .detectors import DetectorConfig, default_detectors
from .models import AnalysisReport, RefactorFinding, RefactorPlan
from .patterns import PATTERN_SPECS
from .planner import build_refactor_plans


def analyze_modules(
    modules: list, config: DetectorConfig | None = None
) -> list[RefactorFinding]:
    config = config or DetectorConfig()
    findings: list[RefactorFinding] = []
    for detector in default_detectors():
        findings.extend(detector.detect(modules, config))
    return sorted(
        findings,
        key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
    )


def analyze_path(
    root: Path, config: DetectorConfig | None = None
) -> list[RefactorFinding]:
    modules = parse_python_modules(root)
    return analyze_modules(modules, config)


def plan_path(root: Path, config: DetectorConfig | None = None) -> list[RefactorPlan]:
    findings = analyze_path(root, config)
    return build_refactor_plans(findings, root)


def _json_payload(
    findings: list[RefactorFinding],
    plans: list[RefactorPlan],
    modules: list,
) -> dict[str, object]:
    report = AnalysisReport(findings=tuple(findings), plans=tuple(plans))
    graph = build_observation_graph(modules)
    payload = report.to_dict()
    payload["observations"] = [asdict(item) for item in graph.observations]
    payload["fibers"] = [asdict(item) for item in graph.fibers]
    return payload


def _format_markdown(
    findings: list[RefactorFinding], plans: list[RefactorPlan] | None = None
) -> str:
    sections: list[str] = []
    if findings:
        sections.append(_format_findings_markdown(findings))
    elif not plans:
        sections.append("No refactoring findings.")
    if plans is not None:
        sections.append(_format_plans_markdown(plans))
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
            f"Pattern {pattern_id.value}" for pattern_id in plan.application_order
        )
        lines.append(f"{index}. {plan.subsystem}")
        lines.append(f"   - Summary: {plan.summary}")
        lines.append(
            f"   - Primary pattern: Pattern {primary.pattern_id.value}: {primary.name}"
        )
        if plan.secondary_pattern_ids:
            secondary = ", ".join(
                f"Pattern {pattern_id.value}: {PATTERN_SPECS[pattern_id].name}"
                for pattern_id in plan.secondary_pattern_ids
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
            "   - Outcome: "
            f"removable LOC {plan.outcome.lower_bound_removable_loc}-{plan.outcome.upper_bound_removable_loc}; "
            f"loci {plan.outcome.loci_of_change_before}->{plan.outcome.loci_of_change_after}; "
            f"mappings {plan.outcome.repeated_mappings_centralized}; "
            f"dispatch {plan.outcome.dispatch_sites_eliminated}; "
            f"registrations {plan.outcome.registration_sites_removed}; "
            f"shared algorithms {plan.outcome.shared_algorithm_sites_centralized}"
        )
        for action in plan.actions:
            lines.append(f"   - Action: {action.kind} -> {action.description}")
            if action.statement_operation and action.statement_sites:
                site_list = ", ".join(
                    f"{item.file_path}:{item.line}" for item in action.statement_sites
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for nominal architecture."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="nominal_refactor_advisor",
        help="Root path to analyze (defaults to nominal_refactor_advisor).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
    )
    parser.add_argument(
        "--include-plans",
        action="store_true",
        help="Also synthesize subsystem-level composed refactor plans.",
    )
    parser.add_argument(
        "--plans-only",
        action="store_true",
        help="Emit only subsystem-level composed refactor plans.",
    )
    parser.add_argument(
        "--min-duplicate-statements",
        type=int,
        default=3,
        help="Minimum statement count for repeated-method detection.",
    )
    parser.add_argument(
        "--min-string-cases",
        type=int,
        default=3,
        help="Minimum string cases for closed-family dispatch detection.",
    )
    parser.add_argument(
        "--min-attribute-probes",
        type=int,
        default=2,
        help="Minimum attribute probes before surfacing a finding.",
    )
    parser.add_argument(
        "--min-builder-keywords",
        type=int,
        default=3,
        help="Minimum keyword count for repeated record-builder detection.",
    )
    parser.add_argument(
        "--min-export-keys",
        type=int,
        default=3,
        help="Minimum key count for repeated export-dict detection.",
    )
    parser.add_argument(
        "--min-registration-sites",
        type=int,
        default=2,
        help="Minimum manual registration sites before surfacing a class-registration finding.",
    )
    parser.add_argument(
        "--min-hardcoded-string-sites",
        type=int,
        default=3,
        help="Minimum repeated semantic string-literal sites before surfacing an SSOT finding.",
    )
    args = parser.parse_args()

    config = DetectorConfig.from_namespace(args)
    root = Path(args.path)
    modules = parse_python_modules(root)
    findings = analyze_modules(modules, config)
    plans = None
    if args.include_plans or args.plans_only:
        plans = build_refactor_plans(findings, root)
    if args.json:
        json_findings = [] if args.plans_only else findings
        print(
            json.dumps(
                _json_payload(json_findings, plans or [], modules),
                indent=2,
            )
        )
    else:
        if args.plans_only:
            print(_format_plans_markdown(plans or []))
        else:
            print(_format_markdown(findings, plans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
