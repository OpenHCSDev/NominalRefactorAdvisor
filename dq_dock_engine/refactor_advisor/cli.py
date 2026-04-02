from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ast_tools import parse_python_modules
from .detectors import DetectorConfig, default_detectors
from .models import RefactorFinding
from .patterns import PATTERN_SPECS


def analyze_path(
    root: Path, config: DetectorConfig | None = None
) -> list[RefactorFinding]:
    config = config or DetectorConfig()
    modules = parse_python_modules(root)
    findings: list[RefactorFinding] = []
    for detector in default_detectors():
        findings.extend(detector.detect(modules, config))
    return sorted(
        findings,
        key=lambda finding: (finding.pattern_id, finding.title, finding.summary),
    )


def _format_markdown(findings: list[RefactorFinding]) -> str:
    if not findings:
        return "No refactoring findings."
    lines: list[str] = []
    for index, finding in enumerate(findings, start=1):
        pattern = PATTERN_SPECS[finding.pattern_id]
        lines.append(f"{index}. {finding.title}")
        lines.append(f"   - Pattern {pattern.pattern_id}: {pattern.name}")
        lines.append(f"   - Summary: {finding.summary}")
        lines.append(f"   - Capability gap: {finding.capability_gap}")
        lines.append(f"   - Prescription: {pattern.prescription}")
        lines.append(f"   - Canonical shape: {pattern.canonical_shape}")
        lines.append(f"   - Why: {finding.why}")
        lines.append(f"   - Relation: {finding.relation_context}")
        lines.append(f"   - Confidence: {finding.confidence}")
        for step in pattern.first_moves:
            lines.append(f"   - First move: {step}")
        for skeleton in pattern.example_skeletons:
            lines.append(f"   - Example skeleton: {skeleton}")
        if finding.scaffold:
            lines.append(f"   - Suggested scaffold: {finding.scaffold}")
        for item in finding.evidence:
            lines.append(f"   - Evidence: {item.file_path}:{item.line} `{item.symbol}`")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AST-driven refactoring advisor for DQ-Dock."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="dq_dock_engine",
        help="Root path to analyze (defaults to dq_dock_engine).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
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
    args = parser.parse_args()

    config = DetectorConfig.from_namespace(args)
    findings = analyze_path(Path(args.path), config)
    if args.json:
        print(json.dumps([finding.to_dict() for finding in findings], indent=2))
    else:
        print(_format_markdown(findings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
