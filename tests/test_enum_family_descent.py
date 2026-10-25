"""Descent replays: CasesFold replays PR #26 (ClassMroViolation -> OpenClassMro
family) from both pre-images; CasesFamily replays AnalysisCacheStatus, an enum that
declares behaviour, into a class family.

The fixtures are copies (CasesFold) or verbatim excerpts (CasesFamily) of NRA source
at the commits named in each fixture, so the replay does not depend on NRA's current
source. Comparison ignores docstrings and formatting: both sides are unparsed,
docstrings dropped, empty and ``...`` bodies read as ``pass``, from-import names
read in sorted order, then Black-formatted.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import black
import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.cases_family import CasesFamily
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodPlanDocument,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.descent import CasesFold, Descent
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    ExternalEnumCaseRecoveryDetector,
)
from nominal_refactor_advisor.json_reports import json_report_object

FIXTURES = Path(__file__).parent / "fixtures" / "enum_family"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
ENUM = "nominal_refactor_advisor.class_mro.ClassMroViolation"
FAMILY_ENUM = "nominal_refactor_advisor.analysis_cache.AnalysisCacheStatus"
FAMILY_FILES = (
    ("nominal_refactor_advisor/analysis_cache.py", "analysis_cache"),
    ("nominal_refactor_advisor/analysis.py", "analysis"),
    ("tests/test_analysis_cache.py", "analysis_cache_tests"),
)
FAMILY_NAMES = ("require_exact_identity", "require_summary_payload")


@pytest.fixture(autouse=True)
def _declared_case_root(tmp_path: Path) -> None:
    """Each staged tree is a repository declaring NRA's own family root."""

    shutil.copy(PYPROJECT, tmp_path / "pyproject.toml")


def _normalized(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            node.names.sort(key=lambda alias: alias.name)
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if not body or (
            len(body) == 1
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and body[0].value.value is Ellipsis
        ):
            node.body = [ast.Pass()]
    return black.format_str(ast.unparse(tree), mode=black.Mode())


def _staged(tmp_path: Path, form: str) -> tuple[CodemodSourceSnapshot, Path, Path]:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    module = package / "class_mro.py"
    tests = tmp_path / "tests" / "test_class_mro.py"
    shutil.copy(FIXTURES / f"class_mro_{form}.py", module)
    shutil.copy(FIXTURES / "class_mro_tests_before.py", tests)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    return snapshot, module, tests


def _target(snapshot: CodemodSourceSnapshot, symbol: str) -> SourceRewriteTarget:
    (target,) = (
        target
        for target in snapshot.source_index.targets_matching_repository_symbol(symbol)
        if target.is_class
    )
    return SourceRewriteTarget(target_id=target.target_id)


def _fold(snapshot: CodemodSourceSnapshot, names: tuple[str, ...]) -> CasesFold:
    return CasesFold(target=_target(snapshot, ENUM), rationale="", names=names)


def _simulated(snapshot: CodemodSourceSnapshot, operation: Descent):
    return CodemodPlanDocument(
        recipes=(RefactorRecipe("fold").with_operation(operation),)
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


@pytest.mark.parametrize(
    ("form", "names"),
    (
        (
            "comparison",
            ("known_prefix", "authority", "single_base_prefix", "HierarchyViolation"),
        ),
        ("flag", ("HierarchyViolation",)),
    ),
)
def test_cases_fold_replays_the_class_mro_family(
    tmp_path: Path, form: str, names: tuple[str, ...]
) -> None:
    snapshot, module, tests = _staged(tmp_path, form)
    operation = _fold(snapshot, names)

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources
    assert _normalized(rewritten[module.as_posix()]) == _normalized(
        (FIXTURES / "class_mro_family.py").read_text()
    )
    assert _normalized(rewritten[tests.as_posix()]) == _normalized(
        (FIXTURES / "class_mro_tests_after.py").read_text()
    )
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )


def test_cases_fold_hands_back_the_unanswered_parent_name(tmp_path: Path) -> None:
    snapshot, module, _tests = _staged(tmp_path, "flag")

    with pytest.raises(ValueError) as refusal:
        _fold(snapshot, ()).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: 1 name(s) needed, 0 given: [0] {module.as_posix()}:131: name of "
        "the parent shared by CyclicHierarchy, InconsistentHierarchy"
    )


@pytest.mark.parametrize(
    ("form", "sites"),
    (
        ("comparison", ["ClassMroAuthority.lookup_prefix"]),
        ("flag", ["OpenClassMro.known_prefix"]),
        ("family", []),
    ),
)
def test_external_enum_case_recovery_sees_the_pre_images_and_not_the_family(
    tmp_path: Path, form: str, sites: list[str]
) -> None:
    _staged(tmp_path, form)

    found = ExternalEnumCaseRecoveryDetector().detect(
        list(parse_python_modules(tmp_path)), DetectorConfig()
    )

    assert [evidence.symbol for finding in found for evidence in finding.evidence] == (
        [ENUM, *sites] if sites else []
    )


def _family_staged(tmp_path: Path, form: str) -> CodemodSourceSnapshot:
    (tmp_path / "nominal_refactor_advisor").mkdir()
    (tmp_path / "nominal_refactor_advisor" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    for path, stem in FAMILY_FILES:
        shutil.copy(FIXTURES / f"{stem}_{form}.py", tmp_path / path)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())


def _family(snapshot: CodemodSourceSnapshot, names: tuple[str, ...]) -> CasesFamily:
    return CasesFamily(target=_target(snapshot, FAMILY_ENUM), rationale="", names=names)


def test_cases_family_replays_the_analysis_cache_status_family(
    tmp_path: Path,
) -> None:
    snapshot = _family_staged(tmp_path, "before")
    operation = _family(snapshot, FAMILY_NAMES)

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources
    for path, stem in FAMILY_FILES:
        assert _normalized(rewritten[(tmp_path / path).as_posix()]) == _normalized(
            (FIXTURES / f"{stem}_after.py").read_text()
        )
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )


def test_cases_family_hands_back_every_unnamed_branch_site(tmp_path: Path) -> None:
    snapshot = _family_staged(tmp_path, "before")

    with pytest.raises(ValueError) as refusal:
        _family(snapshot, FAMILY_NAMES[:1]).source_edits_from_snapshot(snapshot)

    package = (tmp_path / "nominal_refactor_advisor").as_posix()
    prompt = "name of the method this branch becomes on the family"
    assert str(refusal.value) == (
        "handback: 2 name(s) needed, 1 given: "
        f"[0] {package}/analysis.py:27: {prompt}; "
        f"[1] {package}/analysis_cache.py:222: {prompt}"
    )


def test_cases_fold_refuses_an_enum_that_declares_behaviour(tmp_path: Path) -> None:
    snapshot = _family_staged(tmp_path, "before")
    operation = CasesFold(target=_target(snapshot, FAMILY_ENUM), rationale="", names=())

    with pytest.raises(ValueError, match="fold moves data, not behaviour"):
        operation.source_edits_from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("form", "symbols"),
    (
        (
            "before",
            [
                FAMILY_ENUM,
                "CachedAnalysisResult.exact_cache_identity",
                "AnalysisFindingSummaryLookup.__post_init__",
            ],
        ),
        ("after", []),
    ),
)
def test_external_enum_case_recovery_sees_the_family_pre_image_only(
    tmp_path: Path, form: str, symbols: list[str]
) -> None:
    _family_staged(tmp_path, form)

    found = ExternalEnumCaseRecoveryDetector().detect(
        list(parse_python_modules(tmp_path)), DetectorConfig()
    )

    assert [
        evidence.symbol for finding in found for evidence in finding.evidence
    ] == symbols


SPLIT_ENUM_SOURCE = """from dataclasses import dataclass
from enum import StrEnum


class Shape(StrEnum):
    def __new__(cls, value: str, flagged: bool) -> "Shape":
        member = str.__new__(cls, value)
        member._value_ = value
        member._flagged = flagged
        return member

    FIRST = ("first", True)
    SECOND = ("second", True)
    THIRD = ("third", False)
    FOURTH = ("fourth", False)

    @property
    def flagged(self) -> bool:
        return self._flagged


@dataclass
class Holder:
    shape: Shape

    def check(self, given: bool) -> None:
        if given is not self.shape.flagged:
            raise ValueError("shape")
"""


def test_cases_family_keeps_the_unfolded_body_when_members_would_share_an_override(
    tmp_path: Path,
) -> None:
    (tmp_path / "split.py").write_text(SPLIT_ENUM_SOURCE)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "split.Shape"), rationale="", names=("check_shape",)
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    (source,) = simulation.simulation.rewritten_sources.values()
    tree = ast.parse(source)
    overriders = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(statement, ast.FunctionDef) and statement.name == "check_shape"
            for statement in node.body
        )
    ]
    assert overriders == ["Shape"]
    assert "if given is not cls.flagged:" in source


ELIF_ENUM_SOURCE = """from dataclasses import dataclass
from enum import StrEnum


class Need(StrEnum):
    def __new__(cls, value: str, required: bool) -> "Need":
        member = str.__new__(cls, value)
        member._value_ = value
        member._required = required
        return member

    REQUIRED = ("required", True)
    FORBIDDEN = ("forbidden", False)

    @property
    def required(self) -> bool:
        return self._required


@dataclass
class Holder:
    need: Need
    reference: str | None

    def check(self) -> None:
        if self.need.required:
            if self.reference is None:
                raise ValueError("missing")
        elif self.reference is not None:
            raise ValueError("unexpected")
"""


def test_cases_family_turns_a_decided_away_elif_into_an_if(tmp_path: Path) -> None:
    (tmp_path / "need.py").write_text(ELIF_ENUM_SOURCE)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "need.Need"), rationale="", names=("check_reference",)
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    (source,) = simulation.simulation.rewritten_sources.values()
    tree = ast.parse(source)
    (forbidden,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"Required", "Forbidden"}
        and any(
            isinstance(statement, ast.FunctionDef)
            and statement.name == "check_reference"
            for statement in node.body
        )
    )
    assert "elif" not in ast.get_source_segment(source, forbidden)


def test_cases_family_gives_each_independent_if_its_own_method(tmp_path: Path) -> None:
    (tmp_path / "pair.py").write_text(
        ELIF_ENUM_SOURCE.replace(
            """            raise ValueError("unexpected")
""",
            """            raise ValueError("unexpected")
        if self.need.required:
            if self.reference == "":
                raise ValueError("empty")
""",
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    with pytest.raises(ValueError) as refusal:
        CasesFamily(
            target=_target(snapshot, "pair.Need"), rationale="", names=()
        ).source_edits_from_snapshot(snapshot)

    prompt = "name of the method this branch becomes on the family"
    module = (tmp_path / "pair.py").as_posix()
    assert str(refusal.value) == (
        f"handback: 2 name(s) needed, 0 given: [0] {module}:26: {prompt}; "
        f"[1] {module}:31: {prompt}"
    )


def test_cases_family_keeps_sorted_imports_sorted(tmp_path: Path) -> None:
    (tmp_path / "need.py").write_text(ELIF_ENUM_SOURCE)
    (tmp_path / "reader.py").write_text(
        "from need import Holder, Need, helper\n\n"
        "assert Need.REQUIRED.required\n"
        "assert Holder and helper\n"
    )
    (tmp_path / "need.py").write_text(
        (tmp_path / "need.py").read_text() + "\n\ndef helper() -> None:\n    pass\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "need.Need"), rationale="", names=("check_reference",)
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources
    family = rewritten[(tmp_path / "need.py").as_posix()]
    assert family.startswith(
        "from dataclasses import dataclass\n"
        "from nominal_refactor_advisor.case_root import Case\n"
    )
    assert rewritten[(tmp_path / "reader.py").as_posix()].startswith(
        "from need import (\n    Holder,\n    Required,\n    helper,\n)\n"
    )


LAMBDA_ENUM_SOURCE = """from collections.abc import Callable
from enum import StrEnum


class Echo(StrEnum):
    def __new__(cls, value: str, say: Callable[[list[str], str], None]) -> "Echo":
        member = str.__new__(cls, value)
        member._value_ = value
        member._say = say
        return member

    LOUD = ("loud", lambda heard, word: heard.append(word.upper()))
    MUTE = ("mute", lambda heard, word: None)

    def speak(self, heard: list[str], word: str) -> None:
        self._say(heard, word)
"""


def test_cases_family_inlines_lambda_data_into_its_only_caller(tmp_path: Path) -> None:
    (tmp_path / "echo.py").write_text(LAMBDA_ENUM_SOURCE)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "echo.Echo"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    (source,) = simulation.simulation.rewritten_sources.values()
    assert "_say" not in source
    tree = ast.parse(source)
    speak = {
        node.name: [
            ast.unparse(statement)
            for function in node.body
            if isinstance(function, ast.FunctionDef) and function.name == "speak"
            for statement in function.body
        ]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert speak == {
        "Echo": ["heard.append(word.upper())"],
        "Loud": [],
        "Mute": ["pass"],
    }


def test_cases_family_moves_a_returning_branch_with_the_rest_of_its_function(
    tmp_path: Path,
) -> None:
    (tmp_path / "tail.py").write_text(
        ELIF_ENUM_SOURCE.replace(
            """    def check(self) -> None:
        if self.need.required:
            if self.reference is None:
                raise ValueError("missing")
        elif self.reference is not None:
            raise ValueError("unexpected")
""",
            """    def label(self) -> str:
        if self.need.required:
            return "required"
        return "free"
""",
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "tail.Need"), rationale="", names=("label",)
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    (source,) = simulation.simulation.rewritten_sources.values()
    tree = ast.parse(source)
    bodies = {
        (node.name, function.name): [ast.unparse(line) for line in function.body]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for function in node.body
        if isinstance(function, ast.FunctionDef) and function.name == "label"
    }
    assert bodies == {
        ("Need", "label"): ["return 'required'"],
        ("Forbidden", "label"): ["return 'free'"],
        ("Holder", "label"): ["return self.need.label()"],
    }


def test_cases_family_follows_a_re_export_to_the_enum(tmp_path: Path) -> None:
    (tmp_path / "need.py").write_text(ELIF_ENUM_SOURCE)
    (tmp_path / "hub.py").write_text(
        "from need import Need as Need\n\nassert Need.FORBIDDEN.value\n"
    )
    (tmp_path / "reader.py").write_text(
        "from hub import Need\n\nassert Need.REQUIRED.required\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "need.Need"), rationale="", names=("check_reference",)
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources
    assert rewritten[(tmp_path / "hub.py").as_posix()] == (
        "from need import (\n    Forbidden,\n    Need as Need,\n)\n\n"
        "assert Forbidden.value\n"
    )
    assert rewritten[(tmp_path / "reader.py").as_posix()] == (
        "from need import Required\n\nassert Required.required\n"
    )


def test_cases_family_hands_back_an_enum_imported_inside_a_function(
    tmp_path: Path,
) -> None:
    (tmp_path / "need.py").write_text(ELIF_ENUM_SOURCE)
    (tmp_path / "reader.py").write_text(
        "def read() -> bool:\n"
        "    from need import Need\n\n"
        "    return Need.REQUIRED.required\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    with pytest.raises(ValueError) as refusal:
        CasesFamily(
            target=_target(snapshot, "need.Need"),
            rationale="",
            names=("check_reference",),
        ).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: {(tmp_path / 'reader.py').as_posix()}:2: enum imported inside "
        "a function; hoist it to module scope"
    )


def test_cases_family_spells_the_case_import_as_its_package_does(
    tmp_path: Path,
) -> None:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper.py").write_text("THING = 1\n")
    (package / "need.py").write_text(
        ELIF_ENUM_SOURCE.replace(
            "from enum import StrEnum\n",
            "from enum import StrEnum\n\nfrom .helper import THING\n",
        )
        + "\n\nassert THING\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "nominal_refactor_advisor.need.Need"),
        rationale="",
        names=("check_reference",),
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[(package / "need.py").as_posix()]
    assert family.startswith(
        "from dataclasses import dataclass\n\n"
        "from .case_root import Case\n"
        "from .helper import THING\n"
    )
    assert "class Need(Case):\n" in family
    assert "    value: str\n" not in family


def test_cases_family_hands_back_a_member_whose_value_is_not_a_string(
    tmp_path: Path,
) -> None:
    (tmp_path / "level.py").write_text(
        "from enum import IntEnum\n\n\n"
        "class Level(IntEnum):\n"
        "    LOW = 1\n"
        "    HIGH = 2\n\n"
        "    @property\n"
        "    def high(self) -> bool:\n"
        "        return self is Level.HIGH\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    with pytest.raises(ValueError) as refusal:
        CasesFamily(
            target=_target(snapshot, "level.Level"), rationale="", names=()
        ).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: {(tmp_path / 'level.py').as_posix()}:5: member LOW's value is "
        "not a string literal; Case carries a str value"
    )


def test_cases_family_keeps_a_string_default_a_value(tmp_path: Path) -> None:
    (tmp_path / "mutation.py").write_text(
        "from enum import Enum\n\n\n"
        "class Mutation(Enum):\n"
        '    INSERT = "INSERT"\n'
        '    DELETE = "DELETE"\n\n'
        '    def trigger(self, *, when: str = "") -> str:\n'
        "        return self.value + when\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "mutation.Mutation"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[
        (tmp_path / "mutation.py").as_posix()
    ]
    assert 'def trigger(cls, *, when: str = "") -> str:' in family


def test_cases_family_root_with_only_methods_has_one_blank_line_after_its_docstring(
    tmp_path: Path,
) -> None:
    (tmp_path / "mutation.py").write_text(
        "from enum import Enum\n\n\n"
        "class Mutation(Enum):\n"
        '    """Row events."""\n\n'
        '    INSERT = "INSERT"\n\n'
        "    def trigger(self) -> str:\n"
        "        return self.value\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "mutation.Mutation"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    family = simulation.simulation.rewritten_sources[
        (tmp_path / "mutation.py").as_posix()
    ]
    assert (
        'class Mutation(Case):\n    """Row events."""\n\n    @classmethod\n' in family
    )


def test_cases_family_root_with_nothing_of_its_own_passes(tmp_path: Path) -> None:
    (tmp_path / "reason.py").write_text(
        "from enum import StrEnum\n\n\n"
        "class Reason(StrEnum):\n"
        '    OBSERVER = "observer"\n\n\n'
        "assert Reason.OBSERVER.value\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "reason.Reason"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[
        (tmp_path / "reason.py").as_posix()
    ]
    assert "class Reason(Case):\n    pass\n" in family
    assert "class Observer(Reason):\n" in family


def test_cases_family_leaves_annotated_metadata_values(tmp_path: Path) -> None:
    (tmp_path / "tier.py").write_text(
        "from enum import StrEnum\n"
        "from typing import Annotated\n\n\n"
        "class Tier(StrEnum):\n"
        '    STANDARD = "standard"\n\n'
        "    @property\n"
        "    def label(self) -> str:\n"
        "        return self.value\n\n\n"
        'def choose(tier: Annotated[Tier, "Tier help text", Tier.STANDARD]) -> str:\n'
        "    return tier.label\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "tier.Tier"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[(tmp_path / "tier.py").as_posix()]
    assert (
        'def choose(tier: Annotated[type[Tier], "Tier help text", Standard]) -> str:'
        in family
    )


def test_cases_family_hands_back_without_a_declared_family_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "plain"\n')
    (tmp_path / "reason.py").write_text(
        "from enum import StrEnum\n\n\n"
        "class Reason(StrEnum):\n"
        '    OBSERVER = "observer"\n'
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())

    with pytest.raises(ValueError) as refusal:
        CasesFamily(
            target=_target(snapshot, "reason.Reason"), rationale="", names=()
        ).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: {tmp_path / 'pyproject.toml'}: declare a family root in"
        " [tool.nominal-refactor-advisor.family-root]"
    )


DERIVED_ROOT_SOURCE = """import re


def words(stem: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", stem).lower()


class Root:
    declared_name: str

    def __init_subclass__(cls, *, declared_name: str | None = None) -> None:
        if Root not in cls.__bases__:
            cls.declared_name = declared_name or words(cls.__name__)

    @classmethod
    def member_named(cls, name: object) -> type:
        (member,) = (m for m in cls.__subclasses__() if m.declared_name == name)
        return member
"""

DERIVED_PYPROJECT = """[tool.nominal-refactor-advisor.family-root]
root = "family:Root"
value = "declared_name"
decoder = "member_named"
derived-value = "family:words"
"""

DERIVED_ENUM_SOURCE = """from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    INSERT = "INSERT"
    PLAIN_OBSERVER = "plain_observer"

    def label(self) -> str:
        return self.value.lower()


@dataclass
class Holder:
    kind: Kind

    def wire(self) -> str:
        return self.kind.value


def decoded(text: str) -> Kind:
    return Kind(text)


def names() -> list[str]:
    return [kind.value for kind in Kind]


def spelled(kind: Kind) -> str:
    return kind.value + Kind.INSERT.value
"""


def _derived_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str):
    (tmp_path / "pyproject.toml").write_text(DERIVED_PYPROJECT)
    (tmp_path / "family.py").write_text(DERIVED_ROOT_SOURCE)
    (tmp_path / "kind.py").write_text(source)
    monkeypatch.syspath_prepend(tmp_path)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())


def test_cases_family_carries_values_through_the_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _derived_staged(tmp_path, monkeypatch, DERIVED_ENUM_SOURCE)
    operation = CasesFamily(
        target=_target(snapshot, "kind.Kind"), rationale="", names=()
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[(tmp_path / "kind.py").as_posix()]
    assert family.startswith(
        "from dataclasses import dataclass\nfrom family import Root\n"
    )
    assert "class Kind(Root):\n" in family
    assert 'class Insert(Kind, declared_name="INSERT"):\n    pass\n' in family
    assert "class PlainObserver(Kind):\n    pass\n" in family
    assert "return cls.declared_name.lower()" in family
    assert "return self.kind.declared_name" in family
    assert "return Kind.member_named(text)" in family
    assert "[kind.declared_name for kind in Kind.__subclasses__()]" in family
    assert "return kind.declared_name + Insert.declared_name" in family
    namespace: dict[str, object] = {}
    exec(compile(family, "kind.py", "exec"), namespace)
    assert namespace["decoded"]("plain_observer") is namespace["PlainObserver"]
    assert namespace["names"]() == ["INSERT", "plain_observer"]
    assert namespace["Holder"](namespace["Insert"]).wire() == "INSERT"


def test_cases_family_hands_back_a_value_read_through_an_ambiguous_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _derived_staged(
        tmp_path,
        monkeypatch,
        DERIVED_ENUM_SOURCE
        + "\n\nclass Other:\n    kind: Holder\n\n\n"
        + "def read(thing) -> str:\n    return thing.kind.value\n",
    )

    with pytest.raises(ValueError) as refusal:
        CasesFamily(
            target=_target(snapshot, "kind.Kind"), rationale="", names=()
        ).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: {(tmp_path / 'kind.py').as_posix()}:38: thing.kind.value: field"
        " kind is declared holding the enum and holding another type that may carry"
        " value"
    )


def test_cases_family_reads_a_typed_field_by_its_declared_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _derived_staged(
        tmp_path,
        monkeypatch,
        DERIVED_ENUM_SOURCE
        + "\n\nclass Other:\n    kind: Holder\n    pairs: tuple[tuple[str, Holder], ...]\n\n"
        + "    def read(self) -> str:\n        return self.kind.value\n\n"
        + "    def wires(self) -> list[str]:\n"
        + "        pairs, other = self.pairs, self\n"
        + "        return [holder.kind.value for _name, holder in pairs]\n",
    )

    simulation = _simulated(
        snapshot,
        CasesFamily(target=_target(snapshot, "kind.Kind"), rationale="", names=()),
    )

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[(tmp_path / "kind.py").as_posix()]
    assert "        return self.kind.value\n" in family
    assert "[holder.kind.declared_name for _name, holder in pairs]" in family


def test_cases_family_refuses_a_member_read_through_a_receiver(tmp_path: Path) -> None:
    (tmp_path / "need.py").write_text(ELIF_ENUM_SOURCE)
    (tmp_path / "reader.py").write_text(
        "from need import Need\n\n\n"
        "def required(kind: type[Need] = Need) -> bool:\n"
        "    return kind.REQUIRED.required\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path), ())
    operation = CasesFamily(
        target=_target(snapshot, "need.Need"), rationale="", names=("check_reference",)
    )

    simulation = _simulated(snapshot, operation)

    (violation,) = json_report_object(simulation)["architecture_guard_report"][
        "violations"
    ]
    assert simulation.is_clean is False
    assert (violation["file_path"], violation["qualname"]) == (
        (tmp_path / "reader.py").as_posix(),
        "required",
    )



DECODED_FROM_TEXT = "def decoded(text: str) -> Kind:\n    return Kind(text)\n"


def _decoded_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder: str
) -> tuple[CodemodSourceSnapshot, CasesFamily]:
    assert DERIVED_ENUM_SOURCE.count(DECODED_FROM_TEXT) == 1
    snapshot = _derived_staged(
        tmp_path, monkeypatch, DERIVED_ENUM_SOURCE.replace(DECODED_FROM_TEXT, decoder)
    )
    return snapshot, CasesFamily(
        target=_target(snapshot, "kind.Kind"), rationale="", names=()
    )


def test_cases_family_decodes_a_value_captured_from_a_text_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, operation = _decoded_by(
        tmp_path,
        monkeypatch,
        "def decoded(row: dict[str, str]) -> Kind:\n"
        "    match row:\n"
        '        case {"kind": text}:\n'
        "            return Kind(text)\n"
        "    raise ValueError(row)\n",
    )

    simulation = _simulated(snapshot, operation)

    assert simulation.is_clean is True
    family = simulation.simulation.rewritten_sources[(tmp_path / "kind.py").as_posix()]
    assert "return Kind.member_named(text)" in family


@pytest.mark.parametrize(
    "decoder",
    [
        "def decoded(text: Kind) -> Kind:\n    return Kind(text)\n",
        "def decoded(row: dict) -> Kind:\n    return Kind(row['kind'])\n",
    ],
    ids=["member-parameter", "untyped-subscript"],
)
def test_cases_family_hands_back_an_enum_call_that_may_pass_a_member_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder: str
) -> None:
    """Enum(x) also passes a member through; the decoder is declared for values."""

    snapshot, operation = _decoded_by(tmp_path, monkeypatch, decoder)

    with pytest.raises(ValueError, match=r"also passes a member through"):
        operation.source_edits_from_snapshot(snapshot)
