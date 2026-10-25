"""CasesFold replays PR #26 (ClassMroViolation -> OpenClassMro family) from both pre-images.

The fixtures are copies of ``class_mro.py`` and ``tests/test_class_mro.py`` at the
commits named in each fixture, so the replay does not depend on NRA's current
source. Comparison ignores docstrings and formatting: both sides are unparsed,
docstrings dropped, empty and ``...`` bodies read as ``pass``, then Black-formatted.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import black
import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodPlanDocument,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.descent import CasesFold
from nominal_refactor_advisor.json_reports import json_report_object

FIXTURES = Path(__file__).parent / "fixtures" / "enum_family"
ENUM = "nominal_refactor_advisor.class_mro.ClassMroViolation"


def _normalized(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
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


def _fold(snapshot: CodemodSourceSnapshot, names: tuple[str, ...]) -> CasesFold:
    (target,) = (
        target
        for target in snapshot.source_index.targets_matching_repository_symbol(ENUM)
        if target.is_class
    )
    return CasesFold(
        target=SourceRewriteTarget(target_id=target.target_id),
        rationale="",
        names=names,
    )


def _simulated(snapshot: CodemodSourceSnapshot, operation: CasesFold):
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
    assert RefactorRecipeOperation.from_json_value(
        json_report_object(operation)
    ) == operation


def test_cases_fold_hands_back_the_unanswered_parent_name(tmp_path: Path) -> None:
    snapshot, module, _tests = _staged(tmp_path, "flag")

    with pytest.raises(ValueError) as refusal:
        _fold(snapshot, ()).source_edits_from_snapshot(snapshot)

    assert str(refusal.value) == (
        f"handback: 1 name(s) needed, 0 given: [0] {module.as_posix()}:131: name of "
        "the parent shared by CyclicHierarchy, InconsistentHierarchy"
    )
