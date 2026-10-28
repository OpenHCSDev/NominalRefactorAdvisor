"""Callable-only edits preserve argument syntax and native evaluation order."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceDeclaredCallTargetOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _plan(
    path: Path,
    expression: str,
    *,
    callee: str = "legacy",
    count: int = 1,
) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallTargetOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="run"),
                callee=SourceRewriteTarget(file_path=path.as_posix(), qualname=callee),
                expression_source=expression,
                selection_count=SelectionCountExpectation(exact=count),
            ),
        )
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    "expression",
    ("replacement", "replacement if True else legacy", "replacement # chosen"),
)
def test_cli_preserves_argument_bytes_comments_and_native_effects(
    tmp_path: Path, newline: str, expression: str
) -> None:
    path = tmp_path / "probe.py"
    arguments = (
        "(\n"
        "        mark('café'),  # positional value\n"
        "        scale=mark(3), # keyword value\n"
        "    )"
    ).replace("\n", newline)
    prefix = (
        "events = []\n"
        "def mark(value):\n    events.append(value)\n    return value\n"
        "def legacy(value, *, scale): return value * scale\n"
        "def replacement(value, *, scale): return value * scale\n"
        "def run():\n    return legacy"
    ).replace("\n", newline)
    suffix = "\nprint(ascii((run(), events)))\n".replace("\n", newline)
    source = prefix + arguments + suffix
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = _plan(path, expression)
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
        input=json.dumps(json_report_object(plan)),
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["applied"]
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    rewritten = path.read_bytes().decode()
    assert arguments in rewritten
    assert rewritten.startswith(prefix.removesuffix("legacy"))
    assert rewritten.endswith(suffix)


@pytest.mark.parametrize("call", ("legacy(legacy(1))", "legacy(1) + legacy(2)"))
def test_one_stage_redirects_differing_and_nested_calls_without_span_overlap(
    tmp_path: Path, call: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "def legacy(value): return value + 1\n"
        "def replacement(value): return value + 1\n"
        f"def run(): return {call}\nprint(run())\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = _plan(path, "replacement", count=2)
    replay = CodemodPlanSequence.from_json_value(
        json.loads(json.dumps(json_report_object(plan)))
    )
    simulation = replay.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    assert simulation.stage_count == 1
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    run = next(node for node in ast.parse(path.read_text()).body if node.name == "run")
    assert all(
        node.func.id == "replacement"
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
    )


@pytest.mark.parametrize(
    "bases,callee", (("Left, Right", "Right.legacy"), ("Right, Left", "Right.legacy"))
)
def test_native_mro_selects_the_declaration_not_a_same_named_method(
    tmp_path: Path, bases: str, callee: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Base:\n    @classmethod\n    def legacy(cls, value): return value + 1\n"
        "class Left(Base): pass\n"
        "class Right(Base):\n    @classmethod\n    def legacy(cls, value): return value + 2\n"
        f"class Child({bases}): pass\n"
        "def replacement(value): return value + 2\n"
        "def run(): return Child.legacy(3), Base.legacy(3)\n"
        "print(run())\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    simulation = _plan(path, "replacement", callee=callee).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    simulation.apply()
    assert "Base.legacy(3)" in path.read_text()
    assert "Child.legacy(3)" not in path.read_text()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize(
    "run_source,error",
    (
        ("def run(legacy): return legacy(1)", "resolved calls|unresolved"),
        ("def run(): return legacy(1) + legacy(2)", "expected exactly"),
        (
            "def run(flag):\n    alias = legacy\n    if flag: alias = replacement\n"
            "    return legacy(1), alias(2)",
            "unresolved",
        ),
    ),
)
def test_callable_edit_inherits_selection_proof_and_failure_atomicity(
    tmp_path: Path, run_source: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "def legacy(value): return value\n"
        "def replacement(value): return value\n" + run_source + "\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match=error):
        _plan(path, "replacement").simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


def test_callable_edit_protects_comments_inside_the_selected_callable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n    @staticmethod\n    def legacy(value): return value\n"
        "def run(): return (Owner # declaration comment\n    .legacy)(1)\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="remove a comment"):
        _plan(path, "Owner.legacy", callee="Owner.legacy").simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize("expression", ("return replacement", "replacement; legacy"))
def test_callable_edit_requires_one_authored_expression(
    tmp_path: Path, expression: str
) -> None:
    path = tmp_path / "probe.py"
    source = "def legacy(value): return value\ndef run(): return legacy(1)\n"
    path.write_text(source, encoding="utf-8", newline="")
    with pytest.raises(SyntaxError):
        _plan(path, expression).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()
