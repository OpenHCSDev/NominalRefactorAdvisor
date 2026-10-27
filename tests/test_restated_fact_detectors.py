"""Restated facts: second copies of a fact that its owner should hold."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    DispatchToPolymorphismOperation,
    FindingRecipeEvaluator,
)
from nominal_refactor_advisor.detectors import StringKindDispatchDetector
from nominal_refactor_advisor.patterns import PatternId

DETECTORS = (
    StringKindDispatchDetector,
)


def _findings(tmp_path: Path, detector: type, sources: dict[str, str]):
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)
    detector_id = detector.effective_detector_id()
    return [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == detector_id
    ]


def _lines(findings) -> list[int]:
    return sorted(
        {location.line for finding in findings for location in finding.evidence}
    )


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_finding_states_fact_and_owner_and_is_certified(detector):
    spec = detector.finding_spec
    assert spec.certification.value == "certified"
    assert spec.confidence.value == "high"
    assert spec.why == detector.restated.why
    assert detector.restated.fact in spec.why
    assert detector.restated.owner in spec.why


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_recipe_exists_exactly_when_an_operation_is_declared(detector):
    operation = detector.restated.operation
    assert issubclass(detector, FindingRecipeEvaluator) is (operation is not None)


# ---- string dispatch on a kind

DISPATCH = (
    "def run(kind):\n"
    "    if kind == 'a':\n        return 1\n"
    "    elif kind == 'b':\n        return 2\n"
    "    elif kind in ('c', 'd'):\n        return 3\n"
    "    return 0\n"
)


def test_one_subject_compared_with_several_strings_is_a_finding(tmp_path: Path):
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": DISPATCH})
    assert finding.pattern_id is PatternId.STRING_KIND_DISPATCH
    assert _lines(finding.evidence and [finding]) == [2, 4, 6]
    assert "`kind` is compared with 4 strings ('a', 'b', 'c', 'd')" in finding.summary
    assert "No codemod" not in finding.summary
    assert "DispatchToPolymorphismOperation" in finding.summary


def test_two_strings_or_two_subjects_are_not_a_family(tmp_path: Path):
    source = (
        "def run(kind, mode):\n"
        "    if kind == 'a':\n        return 1\n"
        "    if kind == 'b':\n        return 2\n"
        "    if mode == 'c':\n        return 3\n"
        "    return mode == 'd'\n"
    )
    assert _findings(tmp_path, StringKindDispatchDetector, {"m": source}) == []


def test_a_nested_function_does_not_pool_its_arms_with_its_parent(tmp_path: Path):
    source = (
        "def outer(kind):\n"
        "    if kind == 'a':\n        return 1\n"
        "    def inner(kind):\n"
        "        return kind == 'b' or kind == 'c'\n"
        "    return inner\n"
    )
    assert _findings(tmp_path, StringKindDispatchDetector, {"m": source}) == []


def test_match_string_patterns_are_arms(tmp_path: Path):
    source = (
        "def run(kind):\n    match kind:\n"
        "        case 'a' | 'b':\n            return 1\n"
        "        case 'c':\n            return 2\n"
    )
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": source})
    assert "3 strings" in finding.summary


def test_literal_return_dispatch_is_wired_to_polymorphism_operation(tmp_path: Path):
    source = (
        "def code(kind, value):\n"
        '    """Render one declared mode."""\n'
        "    if kind == 'a':\n        return value + 1\n"
        "    elif kind == 'b':\n        return value + 2\n"
        "    elif kind == 'c':\n        return value + 3\n"
        "    raise ValueError(kind)\n"
    )
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": source})
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), (finding,)
    )
    evaluation = StringKindDispatchDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    operations = evaluation.executable_recipe.operations
    assert [type(operation) for operation in operations] == [
        DispatchToPolymorphismOperation
    ]


# ---- attribute access by name


# ---- exact key sets


# ---- foreign absence probes


# ---- legacy markers


# ---- dead modules


# ---- forwarding facades


