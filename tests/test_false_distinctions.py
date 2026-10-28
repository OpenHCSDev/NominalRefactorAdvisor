"""False distinctions: each shape is found, and its look-alikes are not."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors import DetectorConfig, IdenticalArmsDetector
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

MODE = "from enum import Enum\n\n\nclass Mode(Enum):\n    FAST = 1\n    SLOW = 2\n\n\n"


def _findings(detector: type, tmp_path: Path, source: str):
    (tmp_path / "m.py").write_text(source)
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False)
        return detector().detect(modules, DetectorConfig())


def test_identical_arms_is_a_certified_restated_fact():
    spec = IdenticalArmsDetector.finding_spec
    assert spec.certification.value == "certified"
    assert spec.pattern_id is PatternId.IDENTICAL_ARMS
    assert IdenticalArmsDetector.restated.fact in spec.why


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
    (finding,) = _findings(IdenticalArmsDetector, tmp_path, MODE + body)
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
    assert _findings(IdenticalArmsDetector, tmp_path, MODE + body) == []
