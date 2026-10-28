"""False distinctions: each shape is found, and its look-alikes are not."""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_identical_arms import CollapseIdenticalArmsOperation
from nominal_refactor_advisor.detectors import DetectorConfig, IdenticalArmsBranchDetector
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

MODE = "from enum import Enum\n\n\nclass Mode(Enum):\n    FAST = 1\n    SLOW = 2\n\n\n"


def _findings(detector: type, tmp_path: Path, source: str):
    (tmp_path / "m.py").write_text(source)
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False)
        return detector().detect(modules, DetectorConfig())


def test_identical_arms_is_a_certified_restated_fact():
    spec = IdenticalArmsBranchDetector.finding_spec
    assert spec.certification.value == "certified"
    assert spec.pattern_id is PatternId.IDENTICAL_ARMS
    assert IdenticalArmsBranchDetector.restated.fact in spec.why


@pytest.mark.parametrize(
    ("body", "line", "tests"),
    (
        (
            "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
            "    else:\n        return 10\n",
            10,
            "`mode is Mode.FAST`",
        ),
        (
            "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
            "    elif mode is Mode.SLOW:\n        return 10\n    else:\n        return 10\n",
            10,
            "`mode is Mode.FAST` / `mode is Mode.SLOW`",
        ),
        (
            "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
            "    else:\n        if mode is Mode.SLOW:\n            return 10\n"
            "        else:\n            return 10\n",
            10,
            "`mode is Mode.FAST` / `mode is Mode.SLOW`",
        ),
        (
            "def budget(mode):\n    return 10 if mode is Mode.FAST else 10\n",
            10,
            "`mode is Mode.FAST`",
        ),
        (
            "def budget(mode):\n    match mode:\n        case Mode.FAST:\n"
            "            return 10\n        case _:\n            return 10\n",
            10,
            "`mode`",
        ),
        (
            "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
            "    return 10\n",
            10,
            "`mode is Mode.FAST`",
        ),
    ),
    ids=("if-else", "elif-ladder", "nested-else-if", "conditional", "match", "fall-through"),
)
def test_identical_arms_found(tmp_path: Path, body: str, line: int, tests: str):
    (finding,) = _findings(IdenticalArmsBranchDetector, tmp_path, MODE + body)
    assert finding.evidence[0].line == line
    assert finding.summary.startswith(
        f"`budget` branches with identical arms: line {line} on {tests}."
    )


@pytest.mark.parametrize(
    "body",
    (
        # The arms differ.
        "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
        "    else:\n        return 20\n",
        # No final else: `if a or b: S` is a different shape.
        "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
        "    elif mode is Mode.SLOW:\n        return 10\n",
        # One arm differs from the rest.
        "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
        "    elif mode is Mode.SLOW:\n        return 20\n    else:\n        return 10\n",
        # The match has no wildcard, so an unmatched value runs no arm.
        "def budget(mode):\n    match mode:\n        case Mode.FAST:\n"
        "            return 10\n        case Mode.SLOW:\n            return 10\n",
        # The arm does not leave the block, so the following statement runs twice.
        "def budget(mode, seen):\n    if mode is Mode.FAST:\n        seen.append(mode)\n"
        "    seen.append(mode)\n",
        "def budget(mode):\n    return 10 if mode is Mode.FAST else 20\n",
    ),
    ids=("different", "no-else", "one-differs", "no-wildcard", "no-exit", "conditional"),
)
def test_distinct_arms_are_not_found(tmp_path: Path, body: str):
    assert _findings(IdenticalArmsBranchDetector, tmp_path, MODE + body) == []


DRIVER = "\n\nprint([budget(mode) for mode in Mode])\n"
COLLAPSIBLE = {
    "if-else": (
        "def budget(mode):\n    if mode is Mode.FAST:\n        return 10\n"
        "    else:\n        return 10\n",
        "def budget(mode):\n    return 10\n",
    ),
    "elif-ladder": (
        "def budget(mode):\n    total = 1\n    if mode is Mode.FAST:\n        total += 9\n"
        "        return total\n    elif mode is Mode.SLOW:\n        total += 9\n"
        "        return total\n    else:\n        total += 9\n        return total\n",
        "def budget(mode):\n    total = 1\n    total += 9\n    return total\n",
    ),
    "single-line-arms": (
        "def budget(mode):\n    if mode is Mode.FAST: return 10\n    else: return 10\n",
        "def budget(mode):\n    return 10\n",
    ),
    "conditional": (
        "def budget(mode):\n    return (10 if isinstance(mode, Mode) else 10) + 1\n",
        "def budget(mode):\n    return (10) + 1\n",
    ),
    "match": (
        "def budget(mode):\n    match mode:\n        case Mode():\n"
        "            return 10\n        case _:\n            return 10\n",
        "def budget(mode):\n    return 10\n",
    ),
    "fall-through": (
        "def budget(mode):\n    if mode is Mode.FAST and not mode.name:\n"
        "        return 10\n    return 10\n",
        "def budget(mode):\n    return 10\n",
    ),
    "nested-scope": (
        "def budget(mode):\n    def inner():\n        if mode is Mode.FAST:\n"
        "            return 10\n        else:\n            return 10\n    return inner()\n",
        "def budget(mode):\n    def inner():\n        return 10\n    return inner()\n",
    ),
}


def _run(path: Path) -> str:
    return subprocess.check_output([sys.executable, str(path)], text=True)


def _evaluation(tmp_path: Path):
    (finding,) = _findings(IdenticalArmsBranchDetector, tmp_path, (tmp_path / "m.py").read_text())
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    )
    return IdenticalArmsBranchDetector().evaluate_recipe_for_finding(finding, snapshot), snapshot


@pytest.mark.parametrize("name", sorted(COLLAPSIBLE))
def test_collapse_replays_to_the_one_arm_with_identical_behaviour(
    tmp_path: Path, name: str
):
    before, after = COLLAPSIBLE[name]
    path = tmp_path / "m.py"
    path.write_text(MODE + before + DRIVER)
    expected = _run(path)
    evaluation, snapshot = _evaluation(tmp_path)
    (operation,) = evaluation.required_recipe.operations
    assert isinstance(operation, CollapseIdenticalArmsOperation)
    simulation = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert path.read_text() == MODE + after + DRIVER
    assert _run(path) == expected
    assert _findings(IdenticalArmsBranchDetector, tmp_path, path.read_text()) == []


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        (
            "def budget(mode):\n    if mode.is_fast():\n        return 10\n"
            "    else:\n        return 10\n",
            "evaluating `mode.is_fast()` can have an effect",
        ),
        (
            "def budget(mode):\n    if mode == Mode.FAST:\n        return 10\n"
            "    else:\n        return 10\n",
            "evaluating `mode == Mode.FAST` can have an effect",
        ),
        (
            "def budget(mode):\n    return 10 if probe(mode) else 10\n",
            "evaluating `probe(mode)` can have an effect",
        ),
        (
            "def budget(mode):\n    match mode:\n        case Mode.FAST:\n"
            "            return 10\n        case _:\n            return 10\n",
            "matching `Mode.FAST` can have an effect",
        ),
        (
            "def budget(mode):\n    if mode is Mode.FAST:\n        # fast path\n"
            "        return 10\n    else:\n        return 10\n",
            "would drop a comment",
        ),
    ),
    ids=("call", "equality", "conditional-call", "value-pattern", "comment"),
)
def test_collapse_refuses_what_could_change_behaviour(
    tmp_path: Path, body: str, reason: str
):
    (tmp_path / "m.py").write_text(MODE + body)
    evaluation, _snapshot = _evaluation(tmp_path)
    assert reason in evaluation.rejection_reason


def test_a_match_whose_patterns_bind_names_is_not_identical_arms(tmp_path: Path):
    body = (
        "def budget(mode, total):\n    match mode:\n        case Mode(value=total):\n"
        "            return total\n        case _:\n            return total\n"
    )
    assert _findings(IdenticalArmsBranchDetector, tmp_path, MODE + body) == []
