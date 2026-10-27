"""A refactor goal that removes what it can and reports every target it leaves (#72)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_class_overlap_codemod import SURFACE

REFUSED = (
    "\n\n@dataclass(frozen=True)\nclass Span:\n"
    "    start: int\n    stop: int\n    step: int\n\n\n"
    "@dataclass(frozen=True)\nclass LabeledSpan:\n"
    "    start: int\n    stop: int\n    step: int\n    label: str\n\n\n"
    "def is_span(value):\n    return isinstance(value, Span)\n"
)


def _goal(root: Path, *extra: str) -> tuple[int, dict]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(root / "pkg"),
            "--json",
            "--codemod-refactor-goal",
            "class_body_composition",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return completed.returncode, json.loads(completed.stdout)


def test_a_goal_with_a_refused_target_stops_partially_and_says_why(
    tmp_path: Path,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "surface.py").write_text(SURFACE + REFUSED)
    plan_path = tmp_path / "plan.json"

    exit_code, report = _goal(tmp_path, "--codemod-plan-out", str(plan_path))

    proof = report["trajectory_proof"]
    assert exit_code == 0
    assert report["stop_reason"] == "partially_achieved"
    assert proof["status"] == "partially_proved"
    assert report["stage_count"] == 1
    (terminal,) = proof["partial_terminals"]
    (group,) = terminal["remaining_targets"]
    assert group["status"] == "rejected_by_safety_check"
    assert group["reason"] == "the verdict refuses the move"
    assert group["finding_ids"] == list(report["final_target_finding_ids"])
    delta = terminal["reaching_stages"][0]["finding_delta"]
    assert delta["confirmed_expected_removed_finding_count"] == 1
    assert delta["fulfilled_expected_removals"] is True
    recipes = json.loads(plan_path.read_text())["stages"][0]["recipes"]
    assert [operation["target_qualname"] for operation in recipes[0]["operations"]] == [
        "FunctionEvidenceLocationsCandidate"
    ]
    assert (package / "surface.py").read_text() == SURFACE + REFUSED


def test_a_goal_with_nothing_executable_still_has_no_terminal(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "surface.py").write_text(
        "from dataclasses import dataclass\n" + REFUSED
    )

    exit_code, report = _goal(tmp_path)

    proof = report["trajectory_proof"]
    assert exit_code == 1
    assert report["stop_reason"] == "no_proved_trajectory"
    assert proof["partial_terminals"] == []
    assert proof["dead_ends"][0]["depth"] == 0
    assert proof["dead_ends"][0]["partial_rejection"] == ""
