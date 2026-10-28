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
    DeadModuleDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    ForwardingFacadeDetector,
    LegacyPathMarkerDetector,
    NamedAttributeDetector,
    RestatedIdentityDetector,
    StringKindDispatchDetector,
)
from nominal_refactor_advisor.patterns import PatternId

DETECTORS = (
    StringKindDispatchDetector,
    NamedAttributeDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    LegacyPathMarkerDetector,
    DeadModuleDetector,
    ForwardingFacadeDetector,
    RestatedIdentityDetector,
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
    operations = detector.restated.operations
    assert issubclass(detector, FindingRecipeEvaluator) is bool(operations)


def test_new_patterns_continue_the_stable_sequence():
    assert [
        PatternId.STRING_KIND_DISPATCH,
        PatternId.ATTRIBUTE_ACCESS_BY_NAME,
        PatternId.EXACT_KEY_SET,
        PatternId.FOREIGN_ABSENCE_PROBE,
        PatternId.LEGACY_PATH_MARKER,
        PatternId.DEAD_MODULE,
        PatternId.FORWARDING_FACADE,
        PatternId.RESTATED_IDENTITY,
    ] == [PatternId(number) for number in range(30, 38)]


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
    (finding,) = _findings(
        tmp_path, NamedAttributeDetector, {"m": "KEY = 'owner'\n\n\n" + source}
    )
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


def test_each_probe_names_its_line_and_the_declared_owner_or_why_not(tmp_path: Path):
    owner = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\nclass Goal:\n    paused_at: int | None = None\n"
    )
    source = (
        "from owner import Goal\n\n\n"
        "def make() -> Goal:\n    return Goal()\n\n\n"
        "def f(goal: Goal | None, other):\n"
        "    made = make()\n"
        "    if goal.paused_at is None:\n        return other.paused_at is None\n"
        "    return made.paused_at is not None\n"
    )
    findings = _findings(
        tmp_path, ForeignAbsenceProbeDetector, {"owner": owner, "m": source}
    )
    by_receiver = {finding.summary.split("`")[1]: finding.summary for finding in findings}
    assert (
        f"{tmp_path / 'm.py'}:10 `goal.paused_at is None`, owner `owner.Goal` "
        f"({tmp_path / 'owner.py'}:5) declares `paused_at: int | None`"
    ) in by_receiver["goal"]
    assert (
        "owner unresolved: the type checker infers `Any`, not one class"
        in by_receiver["other"]
    )
    assert (
        f"owner `owner.Goal` ({tmp_path / 'owner.py'}:5) declares "
        "`paused_at: int | None` (the type checker infers `Goal`)"
    ) in by_receiver["made"]


_INFERRED_PROBE = (
    "from owner import Goal\n\n\n"
    "def make() -> Goal:\n    return Goal()\n\n\n"
    "def f() -> bool:\n    made = make()\n    return made.paused_at is None\n"
)
_GOAL = "class Goal:\n    paused_at: int | None = None\n"


def test_a_type_name_two_scanned_classes_share_is_refused(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        ForeignAbsenceProbeDetector,
        {"owner": _GOAL, "other_owner": _GOAL, "m": _INFERRED_PROBE},
    )
    assert "owner unresolved: type name `Goal` binds to 2 scanned classes" in (
        finding.summary
    )


def test_without_mypy_the_owner_says_so(tmp_path: Path, monkeypatch):
    from nominal_refactor_advisor import type_checker_session

    monkeypatch.setattr(
        type_checker_session.importlib.util, "find_spec", lambda name: None
    )
    (finding,) = _findings(
        tmp_path, ForeignAbsenceProbeDetector, {"owner": _GOAL, "m": _INFERRED_PROBE}
    )
    assert "owner unresolved: type checker: mypy is not installed" in finding.summary


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


def _dead(tmp_path: Path, sources: dict[str, str]) -> list[str]:
    return sorted(
        finding.evidence[0].symbol
        for finding in _findings(tmp_path, DeadModuleDetector, sources)
    )


def test_a_module_nothing_imports_is_dead(tmp_path: Path):
    sources = {
        "used": "VALUE = 1\n",
        "user": "import used\n\n\ndef f():\n    return used.VALUE\n",
        "orphan": "def lonely():\n    return 1\n",
    }
    assert _dead(tmp_path, sources) == ["orphan", "user"]


def test_modules_that_never_import_each_other_are_standalone_not_dead(tmp_path: Path):
    sources = {"one": "VALUE = 1\n", "two": "VALUE = 2\n"}
    assert _dead(tmp_path, sources) == []


def test_imported_named_main_guarded_and_declared_modules_are_reached(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[project.scripts]\ncli = "pkg.declared:main"\n'
        '\n[project.entry-points."plugins"]\nextra = "pkg.plugged:load"\n'
    )
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    sources = {
        "declared": "def main():\n    return 0\n",
        "plugged": "def load():\n    return 0\n",
        "guarded": "def main():\n    return 0\n\n\nif __name__ == '__main__':\n    main()\n",
        "loaded": "VALUE = 1\n",
        "loader": "import importlib\nMODULE = importlib.import_module('pkg.loaded')\n",
        "anchor": "import pkg.guarded\n",
    }
    for name, source in sources.items():
        (package / f"{name}.py").write_text(source)
    dead = [
        finding.evidence[0].symbol
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    ]
    assert dead == ["pkg.anchor", "pkg.loader"]


def test_a_loader_that_lists_a_package_reaches_every_module_in_it(tmp_path: Path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "loader.py").write_text(
        "import pkgutil\n\n\ndef load(path):\n    return list(pkgutil.iter_modules(path))\n"
    )
    (package / "verb.py").write_text("VALUE = 1\n")
    (tmp_path / "outside.py").write_text("VALUE = 1\n")
    (tmp_path / "anchor.py").write_text(
        "import outside\n\n\nif __name__ == '__main__':\n    print(outside.VALUE)\n"
    )
    (tmp_path / "orphan.py").write_text("VALUE = 1\n")
    dead = [
        finding.evidence[0].symbol
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    ]
    assert dead == ["orphan"]


# ---- forwarding facades

FACADE = (
    "class Inner:\n    pass\n\n\n"
    "class Facade:\n"
    "    def __init__(self, inner: Inner):\n        self._inner = inner\n"
    "    @property\n    def size(self):\n        return self._inner.size\n"
    "    @property\n    def label(self):\n        return self._inner.name\n"
    "    def run(self, a, *rest, mode=None, **extra):\n"
    "        return self._inner.run(a, *rest, mode=mode, **extra)\n"
)


def test_members_that_only_forward_to_a_held_object_are_a_facade(tmp_path: Path):
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": FACADE})
    assert finding.pattern_id is PatternId.FORWARDING_FACADE
    assert "`Facade` holds `self._inner` (of class `Inner`)" in finding.summary
    assert "size -> _inner.size" in finding.summary
    assert "label -> _inner.name" in finding.summary
    assert "run -> _inner.run()" in finding.summary
    assert [location.symbol for location in finding.evidence] == [
        "Facade",
        "Facade.size",
        "Facade.label",
        "Facade.run",
    ]
    assert finding.metrics.forwarded_member_count == 3
    assert finding.metrics.impact_delta.lower_bound_removable_loc == 3
    assert "No codemod operation performs this factoring yet" in finding.summary


def test_a_constructed_held_object_names_its_constructor(tmp_path: Path):
    source = (
        "class Facade:\n"
        "    def __init__(self):\n        self.held = build_inner()\n"
        "    @property\n    def a(self):\n        return self.held.a\n"
        "    @property\n    def b(self):\n        return self.held.b\n"
    )
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": source})
    assert "`build_inner(...)`" in finding.summary


def test_a_member_that_adds_behaviour_is_not_a_forward(tmp_path: Path):
    source = (
        "class Facade:\n"
        "    def __init__(self, inner):\n        self._inner = inner\n"
        "    @property\n    def a(self):\n        return self._inner.a\n"
        "    @property\n    def doubled(self):\n        return self._inner.a * 2\n"
        "    def run(self, a):\n        return self._inner.run(a + 1)\n"
        "    def checked(self, a):\n        if a:\n            return self._inner.run(a)\n"
        "    def reordered(self, a, b):\n        return self._inner.run(b, a)\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_two_forwards_to_two_different_held_objects_are_not_a_facade(tmp_path: Path):
    source = (
        "class Pair:\n"
        "    def __init__(self, left, right):\n        self.left = left\n        self.right = right\n"
        "    @property\n    def a(self):\n        return self.left.a\n"
        "    @property\n    def b(self):\n        return self.right.b\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_forwarding_to_an_attribute_the_class_never_holds_is_not_a_facade(
    tmp_path: Path,
):
    source = (
        "class Mixin:\n"
        "    @property\n    def a(self):\n        return self.inner.a\n"
        "    @property\n    def b(self):\n        return self.inner.b\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


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


def test_an_annotated_held_class_is_named_without_its_metadata(tmp_path: Path):
    source = (
        "from typing import Annotated\n\n\nclass Facade:\n"
        "    inner: Annotated[Inner, 'wire']\n"
        "    @property\n    def a(self):\n        return self.inner.a\n"
        "    @property\n    def b(self):\n        return self.inner.b\n"
    )
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": source})
    assert "(of class `Inner`)" in finding.summary


# ---- restated identity

IDENTITY_FAMILY = (
    "class Outcome:\n    pass\n\n\n"
    "class Complete(Outcome, declared_name='complete'):\n    pass\n\n\n"
    "class Failed(Outcome, declared_name='failed'):\n"
    "    def label(self):\n        return self.declared_name\n"
)


def test_a_quoted_class_name_and_a_passed_class_name_restate_the_class(tmp_path: Path):
    source = (
        "class TurnItem:\n    pass\n\n\n"
        "def first(items: list['TurnItem']) -> 'TurnItem':\n    return items[0]\n\n\n"
        "def declared():\n    return check('TurnItem', 'Item')\n"
    )
    first, declared = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert first.pattern_id is PatternId.RESTATED_IDENTITY
    assert "first spells class `TurnItem` 2 time(s)" in first.summary
    assert "Declared at `TurnItem` (" in first.summary and "m.py:1)" in first.summary
    assert "declared spells class `TurnItem` 1 time(s)" in declared.summary
    assert "UnquoteResolvedTypeNamesOperation" in first.summary


def test_a_compared_family_spelling_restates_the_member(tmp_path: Path):
    source = IDENTITY_FAMILY + (
        "\n\ndef finished(outcome):\n    return outcome.declared_name == 'complete'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert (
        "`declared_name` spelling of a family member 'complete' (`Complete`)"
        in finding.summary
    )


def test_a_sentinel_class_constant_compared_elsewhere_restates_the_class(
    tmp_path: Path,
):
    source = (
        "class Alpha:\n    sigma = 'alpha'\n\n\n"
        "class Beta:\n    sigma = 'beta'\n\n\n"
        "class Gamma(Alpha):\n    sigma = 'gamma'\n\n\n"
        "class Delta(Alpha):\n    sigma = 'delta'\n\n\n"
        "def choose(obj):\n    if obj.sigma == 'gamma':\n        return 1\n    return 2\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'gamma' (`Gamma`)" in finding.summary


def test_shared_constants_one_word_names_and_foreign_strings_are_not_identities(
    tmp_path: Path,
):
    source = (
        "class Read:\n    pass\n\n\n"
        "class Small(Read):\n    unit = 'kb'\n\n\n"
        "class Large(Read):\n    unit = 'kb'\n\n\n"
        "def f(x, path):\n"
        "    if x.unit == 'kb' or path.suffix == '.jsonl':\n        return send('Read')\n"
        "    return x.status == 'completed'\n"
    )
    assert _findings(tmp_path, RestatedIdentityDetector, {"m": source}) == []


def test_a_member_spelling_passed_as_text_is_flattened_but_its_own_read_is_not(
    tmp_path: Path,
):
    source = (
        IDENTITY_FAMILY
        + "\n\ndef emit(send):\n    return send(Complete.declared_name)\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "read off `Complete` as text" in finding.summary
    assert 15 in _lines([finding])


def test_a_module_name_given_to_an_importer_or_run_flag_restates_the_module(
    tmp_path: Path,
):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "worker.py").write_text("X = 1\n")
    source = (
        "import sys\nfrom importlib import import_module\n\n\n"
        "def load():\n    import_module('pkg.worker')\n"
        "    return (sys.executable, '-m', 'pkg.worker', 'pkg.absent')\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"pkg/run": source})
    assert "load spells module `pkg.worker` 2 time(s)" in finding.summary


def test_a_multiword_spelling_of_a_keyword_family_member_restates_it_anywhere(
    tmp_path: Path,
):
    source = (
        "from typing import ClassVar\n\n\n"
        "class Family:\n"
        "    def __init_subclass__(cls, *, declared_name=None):\n"
        "        super().__init_subclass__()\n\n\n"
        "class ClaimedWrite(Family):\n"
        "    declared_name: ClassVar[str] = 'claimed_write'\n\n\n"
        "def decode(name):\n"
        "    if name != 'claimed_write':\n        raise ValueError(name)\n"
        "    return name == 'main'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'claimed_write' (`ClaimedWrite`)" in finding.summary
    assert "line 14" in finding.summary and "'main'" not in finding.summary


def test_a_classvar_slot_declares_the_spelling_every_member_fills(tmp_path: Path):
    source = (
        "from typing import ClassVar\n\n\n"
        "class ToolCall:\n    declared_name: ClassVar[str]\n\n\n"
        "class SelectedWrite(ToolCall):\n"
        "    declared_name: ClassVar[str] = 'selected_write'\n\n\n"
        "class Row:\n    def __init__(self):\n        self.created_at = 0\n\n\n"
        "def decode(name, row):\n"
        "    index(('selected_write', 'created_at'))\n"
        "    return name != 'selected_write'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'selected_write' (`SelectedWrite`)" in finding.summary
    assert "1 time(s)" in finding.summary


def test_a_quote_the_position_needs_is_not_a_restatement(tmp_path: Path):
    eager = (
        "from typing import ClassVar, TypeVar\n\n"
        "Row = TypeVar('Row', bound='Table')\n\n\n"
        "class Table:\n"
        "    registry: ClassVar[dict[str, type['Table']]] = {}\n\n"
        "    def copy(self) -> 'Table':\n        return self\n"
    )
    assert _findings(tmp_path, RestatedIdentityDetector, {"m": eager}) == []
    (tmp_path / "m.py").unlink()
    lazy = "from __future__ import annotations\n\n" + eager
    findings = _findings(tmp_path, RestatedIdentityDetector, {"lazy": lazy})
    assert sorted(finding.summary.split(" spells")[0] for finding in findings) == [
        "Table",
        "Table.copy",
    ]


@pytest.mark.parametrize(
    ("requires_python", "found"), ((">=3.14,<3.15", 1), (">=3.11", 0))
)
def test_a_quoted_type_checking_name_is_restated_where_annotations_are_lazy(
    tmp_path: Path, requires_python: str, found: int
):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "p"\nrequires-python = "{requires_python}"\n'
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "threads.py").write_text("class Thread:\n    pass\n")
    source = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from .threads import Thread\n\n\n"
        "def resolve(thread: 'Thread') -> None:\n    return None\n"
    )
    findings = _findings(tmp_path, RestatedIdentityDetector, {"pkg/load": source})
    assert len(findings) == found
