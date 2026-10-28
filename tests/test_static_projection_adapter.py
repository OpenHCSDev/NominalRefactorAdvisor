"""Static projection calls become adapter property reads, behaviour intact."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import static_projection_fixture  # noqa: F401  (the projector must be loaded)
import static_projection_reexport  # noqa: F401
from nominal_refactor_advisor.class_index import ParsedModuleSet
from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ProjectStaticCallsToAdapterOperation,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.detectors._runtime import (
    StaticProjectionAdapterCallDetector,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_index import AstTargetNodeKind

PROJECTOR = "static_projection_fixture.LegacyExpressionProjection"
FAMILY = "nominal_refactor_advisor.ast_projection.ExpressionReference"
DETECTOR_ID = "static_projection_adapter_call"
SOURCE = """\
import ast

from static_projection_fixture import LegacyExpressionProjection


def names(source):
    return [
        (
            LegacyExpressionProjection.terminal_name(node),
            LegacyExpressionProjection.qualified_name(node),
        )
        for node in ast.walk(ast.parse(source))
    ]


def nested(node):
    return LegacyExpressionProjection.identifier(node or None)


print(names("a.b[c]\\nx\\nf()"), nested(ast.parse("x", mode="eval").body))
"""


def _operation(path: Path) -> ProjectStaticCallsToAdapterOperation:
    snapshot = CodemodSourceSnapshot.from_modules(ParsedModuleSet.of(parse_python_modules(path.parent)))
    (target_id,) = SourceIndexTargetSelector(
        node_kinds=(AstTargetNodeKind.MODULE,), file_paths=(path.as_posix(),)
    ).target_ids(snapshot)
    return ProjectStaticCallsToAdapterOperation(
        target=SourceRewriteTarget(target_id=target_id),
        projector=PROJECTOR,
        family=FAMILY,
    )


def _simulate(tmp_path: Path, operation: ProjectStaticCallsToAdapterOperation):
    return CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(ParsedModuleSet.of(parse_python_modules(tmp_path)))
    )


def _run(path: Path) -> str:
    here = Path(__file__).parent
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(here.parent), str(here))),
    }
    return subprocess.check_output(
        [sys.executable, str(path)], text=True, env=environment
    )


def test_calls_become_property_reads_and_run_identically(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(SOURCE, encoding="utf-8")
    expected = _run(path)
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    after = path.read_text(encoding="utf-8")
    assert "ExpressionReference.decoded(node).terminal_name" in after
    assert "ExpressionReference.decoded(node or None).identifier" in after
    assert "LegacyExpressionProjection" not in after
    assert "import ExpressionReference" in after
    assert _run(path) == expected


def test_nested_calls_rewrite_together(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "from static_projection_fixture import LegacyExpressionProjection as P\n"
        "from static_projection_fixture import LegacyExpressionProjection\n\n\n"
        "def f(node):\n"
        "    return LegacyExpressionProjection.qualified_name(\n"
        "        LegacyExpressionProjection.identifier(node) and node\n"
        "    )\n",
        encoding="utf-8",
    )
    simulation = _simulate(tmp_path, _operation(path))
    assert simulation.is_clean
    simulation.apply()
    assert (
        "ExpressionReference.decoded(\n        ExpressionReference.decoded(node).identifier and node\n    ).qualified_name"
        in path.read_text(encoding="utf-8")
    )


def test_a_rebound_projector_name_is_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "from static_projection_fixture import LegacyExpressionProjection\n\n\n"
        "def f(LegacyExpressionProjection, node):\n"
        "    return LegacyExpressionProjection.identifier(node)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No use"):
        _simulate(tmp_path, _operation(path))


def test_the_wrong_family_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(SOURCE, encoding="utf-8")
    wrong = ProjectStaticCallsToAdapterOperation(
        target=_operation(path).target,
        projector=PROJECTOR,
        family="nominal_refactor_advisor.foreign_family.AssignmentStatement",
    )
    with pytest.raises(ValueError, match="not a loaded static projection"):
        _simulate(tmp_path, wrong)


def test_operation_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(SOURCE, encoding="utf-8")
    operation = _operation(path)
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )


def test_finding_recipe_rewrites_the_evidenced_module(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(SOURCE, encoding="utf-8")
    (finding,) = [
        item for item in analyze_path(tmp_path) if item.detector_id == DETECTOR_ID
    ]
    assert sorted(item.line for item in finding.evidence) == [9, 10, 17]
    snapshot = CodemodSourceSnapshot.from_modules(ParsedModuleSet.of(parse_python_modules(tmp_path)))
    evaluation = StaticProjectionAdapterCallDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    (operation,) = evaluation.executable_recipe.operations
    assert (operation.projector, operation.family) == (PROJECTOR, FAMILY)
    simulation = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert [
        item for item in analyze_path(tmp_path) if item.detector_id == DETECTOR_ID
    ] == []


def test_reexport_through_a_loaded_module_is_rewritten(tmp_path: Path) -> None:
    user = tmp_path / "user.py"
    user.write_text(
        "from static_projection_reexport import LegacyExpressionProjection\n\n\n"
        "def f(node):\n    return LegacyExpressionProjection.identifier(node)\n",
        encoding="utf-8",
    )
    _simulate(tmp_path, _operation(user)).apply()
    after = user.read_text(encoding="utf-8")
    assert "ExpressionReference.decoded(node).identifier" in after
    assert "LegacyExpressionProjection" not in after


def test_an_import_another_module_takes_from_here_is_kept(tmp_path: Path) -> None:
    hub = tmp_path / "hub.py"
    hub.write_text(
        "from static_projection_fixture import LegacyExpressionProjection\n",
        encoding="utf-8",
    )
    (tmp_path / "user.py").write_text(
        "from hub import LegacyExpressionProjection\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="No use"):
        _simulate(tmp_path, _operation(hub))
    (tmp_path / "user.py").write_text("x = 1\n", encoding="utf-8")
    _simulate(tmp_path, _operation(hub)).apply()
    assert "LegacyExpressionProjection" not in hub.read_text(encoding="utf-8")
