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
ENUM = "nominal_refactor_advisor.class_mro.ClassMroViolation"
FAMILY_ENUM = "nominal_refactor_advisor.analysis_cache.AnalysisCacheStatus"
FAMILY_FILES = (
    ("nominal_refactor_advisor/analysis_cache.py", "analysis_cache"),
    ("nominal_refactor_advisor/analysis.py", "analysis"),
    ("tests/test_analysis_cache.py", "analysis_cache_tests"),
)
FAMILY_NAMES = ("require_exact_identity", "require_summary_payload")


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
    assert family.index("from abc import ABC\n") < family.index("from dataclasses")
    assert rewritten[(tmp_path / "reader.py").as_posix()].startswith(
        "from need import (\n    Holder,\n    Required,\n    helper,\n)\n"
    )
