"""Assignment edits own one statement, not its line or neighbouring bindings."""

from pathlib import Path
import json
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceModuleAssignmentOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def test_module_assignment_preserves_same_line_neighbour(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "value = 1; neighbour = 3  # retain\n", encoding="utf-8", newline=""
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceModuleAssignmentOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                source="value = 2",
            ),
        )
    )
    plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert path.read_text(encoding="utf-8") == "value = 2; neighbour = 3  # retain\n"


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("scope", ("Owner", "run", "run.Owner", "Owner.run"))
def test_cli_replaces_one_assignment_in_a_named_scope(
    tmp_path: Path, newline: str, scope: str
) -> None:
    path = tmp_path / "probe.py"
    headers = {
        "Owner": "class Owner:\n",
        "run": "def run():\n",
        "run.Owner": "def run():\n    class Owner:\n",
        "Owner.run": "class Owner:\n    async def run(self):\n",
    }
    indentation = "    " * len(scope.split("."))
    old = "value = 1; neighbour = 'café'  # retained"
    source = (headers[scope] + indentation + old).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceScopeAssignmentOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname=scope),
                assignment_name="value",
                source="value = 2",
            ),
        )
    )
    result = subprocess.run(
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
        input=json.dumps(json_report_object(plan)),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    assert path.read_bytes() == source.replace("value = 1", "value = 2").encode()


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_assignment_replacement_preserves_multiline_literal_bytes(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = "class Owner:\n    value = None  # retained\nprint(repr(Owner.value))\n"
    path.write_text(source.replace("\n", newline), encoding="utf-8", newline="")
    payload = "value = (\n    '''first\n  literal indent\nlast'''\n)".replace(
        "\n", newline
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceScopeAssignmentOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                assignment_name="value",
                source=payload,
            ),
        )
    )
    plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert subprocess.check_output(
        [sys.executable, str(path)], text=True
    ).strip() == repr("first\n  literal indent\nlast")
    assert b"# retained" in path.read_bytes()


@pytest.mark.parametrize(
    "original,error",
    (
        ("value = 1\nvalue = 2", "ambiguous"),
        ("value = neighbour = 1", "unselected names"),
        ("value = obj.other = 1", "non-name target"),
        ("neighbour = 1", "No assignment"),
        ("value = (\n    1  # explanation\n)", "discard comments"),
    ),
)
def test_module_replacement_rejects_incomplete_statement_ownership(
    tmp_path: Path, original: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(original, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceModuleAssignmentOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                source="value = 3",
            ),
        )
    )
    with pytest.raises(ValueError, match=error):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == original.encode()


@pytest.mark.parametrize(
    "payload",
    (
        "value = 1; other = 2",
        "value = other = 1",
        "obj.value = 1",
        "value += 1",
        "value: (",
        "value = 1 # would swallow neighbour",
        "# lost comment\nvalue = 1",
        "value, other = 1, 2",
    ),
)
def test_scope_replacement_rejects_non_assignment_payloads(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "probe.py"
    source = "class Owner: value: int; neighbour = 3\n"
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceScopeAssignmentOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                assignment_name="value",
                source=payload,
            ),
        )
    )
    with pytest.raises(ValueError):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


def test_sequential_assignment_rename_resolves_the_new_field(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "class Owner: callee: str; neighbour = 3\n", encoding="utf-8", newline=""
    )
    target = SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceScopeAssignmentOperation(
                target=target, assignment_name="callee", source="resolved_target: str"
            ),
            ReplaceScopeAssignmentOperation(
                target=target,
                assignment_name="resolved_target",
                source="resolved_target: str = 'retained'",
            ),
        )
    )
    plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert (
        path.read_text(encoding="utf-8")
        == "class Owner: resolved_target: str = 'retained'; neighbour = 3\n"
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_cli_composes_retained_authority_field_and_consumer_edits(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "resolution.py"
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Target:\n"
        "    declaration: str\n"
        "    access: str = 'direct'\n"
        "@dataclass(frozen=True)\n"
        "class Call:\n"
        "    callee: str\n"
        "    @property\n"
        "    def target_resolution(self) -> Target:\n"
        "        return Target(self.callee)\n"
        "def make(target: Target) -> Call:\n"
        "    return Call(target.declaration)\n"
        "target = Target('Owner.consume', 'instance')\n"
        "call = make(target)\n"
        "print(call.callee, call.target_resolution.declaration)\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    example = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/retain_resolution_refactor.py"
    )
    plan = runpy.run_path(str(example))["PLAN"]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        cwd=tmp_path,
        input=json.dumps(json_report_object(plan)),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from resolution import target, call; "
            "assert call.target_resolution is target; "
            "assert call.target_resolution.access == 'instance'",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def test_assignment_operations_share_nominal_source_authority() -> None:
    from nominal_refactor_advisor import codemod_assignment_operations as owner

    for operation in (
        ReplaceModuleAssignmentOperation,
        ReplaceScopeAssignmentOperation,
    ):
        assert (
            operation.source_edits_from_snapshot
            is owner.AssignmentReplacementOperationABC.source_edits_from_snapshot
        )
        assert operation is vars(owner)[operation.__name__]


def test_recorded_assignment_projection_plan_preserves_native_results(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nominal_refactor_advisor/assignment_projection.py"
    path.parent.mkdir()
    source = (
        "import ast\n"
        "from dataclasses import dataclass\n"
        "from nominal_refactor_advisor.assignment_projection import AssignmentStatementNameProjection\n"
        "@dataclass(frozen=True)\n"
        "class SingleAssignmentAndValueNameProjection:\n"
        "    statement: ast.stmt\n"
        "    @property\n"
        "    def pair(self):\n"
        "        node = self.statement\n"
        "        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):\n"
        "            return node.targets[0].id, node.value\n"
        "        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:\n"
        "            return node.target.id, node.value\n"
        "        return None\n"
        "for text in ('x = 3', 'x: int = 3', 'x: int', 'x += 3', 'a = b = 1', 'obj.x = 3', '(x,) = values'):\n"
        "    result = SingleAssignmentAndValueNameProjection(ast.parse(text).body[0]).pair\n"
        "    print(None if result is None else (result[0], ast.dump(result[1])))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    example = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/assignment_projection_refactor.py"
    )
    plan = runpy.run_path(str(example))["PLAN"]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        cwd=tmp_path,
        input=json.dumps(json_report_object(plan)),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    assert "statement: ast.stmt" not in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
