"""Restated facts: second copies of a fact that its owner should hold."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    DispatchToPolymorphismOperation,
    EraseDeadCompatibilityOperation,
    FindingRecipeEvaluator,
)
from nominal_refactor_advisor.detectors import (
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    LegacyPathMarkerDetector,
    NamedAttributeDetector,
    StringKindDispatchDetector,
)
from nominal_refactor_advisor.patterns import PatternId

DETECTORS = (
    StringKindDispatchDetector,
    NamedAttributeDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    LegacyPathMarkerDetector,
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


def test_attribute_access_by_literal_name_or_with_default_is_a_finding(tmp_path: Path):
    source = (
        "def f(obj):\n"
        "    if hasattr(obj, 'name'):\n        return getattr(obj, 'size', 0)\n"
        "    setattr(obj, 'seen', True)\n    delattr(obj, 'seen')\n"
        "    return getattr(obj, KEY, None)\n"
    )
    (finding,) = _findings(tmp_path, NamedAttributeDetector, {"m": "KEY = 'owner'\n\n\n" + source})
    assert finding.pattern_id is PatternId.ATTRIBUTE_ACCESS_BY_NAME
    assert _lines([finding]) == [5, 6, 7, 8, 9]
    assert "No codemod operation performs this factoring yet" in finding.summary


def test_dynamic_names_without_a_default_and_plain_reads_are_not_findings(
    tmp_path: Path,
):
    source = (
        "def f(obj, name):\n"
        "    setattr(obj, name, 1)\n    return getattr(obj, name, None) or obj.size\n"
    )
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


def test_a_probe_of_an_imported_module_is_platform_detection_not_a_type_fact(
    tmp_path: Path,
):
    source = "import os\n\n\ndef f():\n    return hasattr(os, 'pidfd_open')\n"
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


def test_a_shadowed_getattr_is_a_method_not_the_builtin(tmp_path: Path):
    source = "def f(registry):\n    return registry.getattr('size', 0)\n"
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


# ---- exact key sets

RECORD = (
    "from dataclasses import dataclass\n\n\n@dataclass\nclass Row:\n"
    "    id: int\n    name: str\n\n\n"
)


def test_mapping_keys_compared_with_a_literal_set_are_a_finding(tmp_path: Path):
    source = (
        "def check(payload):\n"
        "    if set(payload) != {'id', 'name'}:\n        raise ValueError\n"
        "    return payload.keys() <= {'id', 'name', 'x'} and payload['id']\n"
    )
    findings = _findings(tmp_path, ExactKeySetDetector, {"m": source})
    assert [_lines([finding]) for finding in findings] == [[2], [4]]
    assert findings[0].pattern_id is PatternId.EXACT_KEY_SET
    assert "Owner: The owner is a record class whose fields are these keys." in (
        findings[0].summary
    )


def test_a_dataclass_with_exactly_those_fields_is_the_resolved_owner(tmp_path: Path):
    source = RECORD + (
        "def check(payload):\n    return set(payload) == {'id', 'name'} and payload.get('id')\n"
    )
    (finding,) = _findings(tmp_path, ExactKeySetDetector, {"m": source})
    assert "`Row`" in finding.summary and "already declares exactly these fields" in (
        finding.summary
    )
    assert finding.authority_evidence is not None
    assert finding.authority_evidence.symbol == "Row"


def test_generator_membership_over_a_literal_is_a_finding(tmp_path: Path):
    source = "def ok(m):\n    return all(k in m for k in ('id', 'name')) and m['id']\n"
    assert len(_findings(tmp_path, ExactKeySetDetector, {"m": source})) == 1


def test_the_characters_or_items_of_a_non_mapping_are_not_a_record(tmp_path: Path):
    source = (
        "def f(text, names):\n"
        "    return set(text) & {'(', '['} or set(names) == {'a', 'b'}\n"
    )
    assert _findings(tmp_path, ExactKeySetDetector, {"m": source}) == []


def test_one_key_or_a_non_literal_side_is_not_a_record(tmp_path: Path):
    source = (
        "KEYS = {'id', 'name'}\n"
        "def check(payload, other):\n"
        "    return set(payload) == {'id'} or set(payload) == KEYS or set(payload) == set(other)\n"
    )
    assert _findings(tmp_path, ExactKeySetDetector, {"m": source}) == []


# ---- foreign absence probes


def test_another_objects_attribute_compared_with_none_is_a_finding(tmp_path: Path):
    source = (
        "def f(goal):\n"
        "    if goal.paused_at is None:\n        return 1\n"
        "    return goal.paused_at is not None\n"
    )
    (finding,) = _findings(tmp_path, ForeignAbsenceProbeDetector, {"m": source})
    assert finding.pattern_id is PatternId.FOREIGN_ABSENCE_PROBE
    assert "`goal` is probed for absence 2 time(s)" in finding.summary


def test_own_state_modules_and_truth_tests_are_not_foreign_probes(tmp_path: Path):
    source = (
        "import sys\n\n\nclass Goal:\n"
        "    def mine(self):\n        return self.paused_at is None\n"
        "    def other_name(this, other):\n        return this.paused_at is None\n"
        "    def truthy(self, other):\n        return not other.paused_at\n"
        "\n\ndef f():\n    return sys.stdout is None\n"
    )
    assert _findings(tmp_path, ForeignAbsenceProbeDetector, {"m": source}) == []


# ---- legacy markers


def test_a_legacy_function_lists_the_consumers_still_on_it(tmp_path: Path):
    findings = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {
            "old": "def load(path):\n    return path\n\n\ndef legacy_load(path):\n    return load(path)\n",
            "user": "from old import legacy_load\n\n\ndef f():\n    return legacy_load('x')\n",
        },
    )
    (finding,) = findings
    assert finding.pattern_id is PatternId.LEGACY_PATH_MARKER
    assert "still use the identifier `legacy_load`" in finding.summary
    assert "user.py" in finding.summary
    assert [location.symbol for location in finding.evidence][-1] == "uses legacy_load"
    assert "EraseDeadCompatibilityOperation" in finding.summary


def test_deprecation_warnings_docstrings_and_parameters_mark_the_old_path(
    tmp_path: Path,
):
    source = (
        "import warnings\n\n\n"
        "def old(path):\n    warnings.warn('x', DeprecationWarning)\n    return path\n\n\n"
        "def documented():\n    '''Deprecated: use new.'''\n\n\n"
        "def f(path, mode=None, legacy_mode=False):\n    return path\n"
    )
    findings = _findings(tmp_path, LegacyPathMarkerDetector, {"m": source})
    assert sorted(finding.evidence[0].symbol for finding in findings) == [
        "documented",
        "f.legacy_mode",
        "old",
    ]


def test_only_the_summary_line_of_a_docstring_declares_the_path_old(tmp_path: Path):
    source = (
        "def parse(path):\n    '''Read a path.\n\n    Replaces the deprecated reader.\n    '''\n"
        "    return path\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_ordinary_declarations_and_plain_imports_of_marked_names_are_not_markers(
    tmp_path: Path,
):
    source = (
        "from elsewhere import legacy_thing\n\n\n"
        "def load(path):\n    '''Read a path.'''\n    return legacy_thing(path)\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_an_unused_legacy_function_gets_an_executable_erasure(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {
            "m": "def load(path):\n    return path\n\n\n"
            "def legacy_load(path):\n    return load(path)\n"
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    (operation,) = evaluation.executable_recipe.operations
    assert isinstance(operation, EraseDeadCompatibilityOperation)


def test_a_marked_parameter_is_not_widened_to_its_function(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {"m": "def f(path, mode=None, legacy_mode=False):\n    return path\n"},
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "is not a function or class declaration" in evaluation.rejection_reason


# ---- dead modules


# ---- forwarding facades


def test_a_legacy_word_with_no_replacement_beside_it_is_not_a_marker(tmp_path: Path):
    source = (
        "class LegacyWord:\n    LEGACY = 'legacy'\n\n\n"
        "def parse_legacy_format(text):\n    return text\n\n\n"
        "def f(path, legacy_mode=False):\n    return path\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_an_alias_of_a_replacement_under_a_legacy_name_is_a_marker(tmp_path: Path):
    source = "def load(path):\n    return path\n\n\nlegacy_load = load\n"
    (finding,) = _findings(tmp_path, LegacyPathMarkerDetector, {"m": source})
    assert "(alias, of load)" in finding.summary


