"""Project a local binding without rewriting its shadows or deleting evaluations."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteFunctionAssignmentsOperation,
    ProjectFunctionLocalOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _operation(path: Path) -> ProjectFunctionLocalOperation:
    return ProjectFunctionLocalOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="render"),
        local_name="candidate",
        projection_source="witness.candidate",
    )


@pytest.mark.parametrize(
    "body",
    (
        "    # candidate stays here\n    return candidate, 'candidate'\n",
        "    return [candidate for candidate in range(candidate)]\n",
        "    return candidate, (lambda candidate: candidate)(2)\n",
        "    def inner():\n        return candidate\n    return inner()\n",
        "    def inner(value=candidate):\n        return value\n    return inner()\n",
        "    class Inner:\n        candidate = 2\n        outer = staticmethod(lambda: candidate)\n    return candidate, Inner.candidate, Inner.outer()\n",
    ),
)
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_local_projection_preserves_initializer_effects_and_shadowed_reads(
    tmp_path: Path, body: str, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from dataclasses import dataclass\n"
        "events = []\n"
        "@dataclass\nclass Witness:\n    candidate: int\n"
        "def initialize(witness):\n    events.append('initialized')\n    return witness.candidate\n"
        "def render(witness):\n    candidate: int = initialize(witness)\n"
        + body
        + "print(render(Witness(3)), events)\n"
    ).replace("\n", newline)
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ast_before = ast.dump(
        snapshot.module_nodes_by_file_path[path.as_posix()], include_attributes=True
    )
    sequence = CodemodPlanSequence.from_operations((_operation(path),))
    replayed = CodemodPlanSequence.from_json_value(
        json.loads(json.dumps(json_report_object(sequence)))
    )
    simulation = replayed.simulate(snapshot)
    assert simulation.is_clean
    assert (
        ast.dump(
            snapshot.module_nodes_by_file_path[path.as_posix()], include_attributes=True
        )
        == ast_before
    )
    simulation.apply()
    assert "candidate: int = initialize(witness)" in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize(
    "body,error",
    (
        ("    candidate = 3\n    candidate = 4\n    return candidate\n", "ambiguous"),
        (
            "    candidate = 3\n    for candidate in range(2): pass\n    return candidate\n",
            "additional bindings",
        ),
        (
            "    candidate = 3\n    def inner():\n        nonlocal candidate\n        candidate = 4\n    return candidate\n",
            "additional bindings",
        ),
        (
            "    before = candidate\n    candidate = 3\n    return candidate\n",
            "before its initializer",
        ),
        (
            "    candidate = candidate + 1\n    return candidate\n",
            "before its initializer",
        ),
        (
            "    if witness:\n        candidate = 3\n    return candidate\n",
            "No assignment",
        ),
        ("    candidate, other = (3, 4)\n    return candidate\n", "unselected names"),
        ("    candidate: int\n    return candidate\n", "with a value"),
        (
            "    candidate = 3\n    return (lambda witness: candidate)(None)\n",
            "captured",
        ),
        (
            "    global candidate\n    candidate = 3\n    return candidate\n",
            "additional bindings",
        ),
    ),
)
def test_local_projection_fails_closed_without_one_unchanged_binding(
    tmp_path: Path, body: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    source = "def render(witness):\n" + body
    path.write_text(source, newline="", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        CodemodPlanSequence.from_operations((_operation(path),)).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_text(encoding="utf-8") == source


def test_local_projection_and_explicit_initializer_removal_apply_through_cli(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Witness:\n    candidate: int\n"
        "def render(witness): candidate = witness.candidate; return candidate + 2\n"
        "print(render(Witness(3)))\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    operation = _operation(path)
    sequence = CodemodPlanSequence.from_operations(
        (
            operation,
            DeleteFunctionAssignmentsOperation(
                target=operation.target, assignment_names=("candidate",)
            ),
        )
    )
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(sequence)),
        text=True,
        capture_output=True,
    )
    assert cli.returncode == 0, cli.stderr
    report = json.loads(cli.stdout)
    assert report["applied"]
    assert report["plan_sequence_simulation"]["stage_count"] == 2
    assert "candidate =" not in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


def test_documented_geometry_batch_projects_reads_and_removes_only_its_local(
    tmp_path: Path,
) -> None:
    example = Path(__file__).parents[1] / "docs/examples/source_geometry_refactor.py"
    plan = runpy.run_path(str(example))["PLAN"]
    path = tmp_path / plan.documents[0].recipes[0].operations[0].target.file_path
    path.parent.mkdir(parents=True)
    source = (
        "from functools import cached_property\n"
        "from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry\n"
        "class ClassBodySourceAuthority:\n"
        "    source = 'class Example: pass\\n'\n"
        "    @cached_property\n    def geometry(self): return SourceTextGeometry(self.source)\n"
        "    def before_first_method_offset(self):\n"
        "        geometry = SourceTextGeometry(self.source)\n"
        "        return geometry.end_offset, geometry.line_offsets\n"
        "print(ClassBodySourceAuthority().before_first_method_offset())\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert "geometry =" not in path.read_text(encoding="utf-8")
    assert (
        path.read_text(encoding="utf-8").count("SourceTextGeometry(self.source)") == 1
    )
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
