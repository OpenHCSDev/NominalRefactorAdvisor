"""Authored function aliases derive their binding and source extent from declarations."""

from pathlib import Path
import json
import subprocess
import sys
import runpy

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    AliasFunctionOperation,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceDeclaredCallOperation,
    ReplaceDeclaredCallArgumentsOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize("class_scope", (False, True))
def test_cli_can_edit_calls_after_introducing_an_alias(
    tmp_path: Path, class_scope: bool
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        "    def first(self, value): return value + 1\n"
        "    def second(self, value): return value + 1\n"
        "    def run(self, value): return self.second(value)\n"
        "print(Owner().run(3))\n"
        if class_scope
        else "def first(value): return value + 1\n"
        "def second(value): return value + 1\n"
        "def run(value): return second(value)\n"
        "print(run(3))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    prefix = "Owner." if class_scope else ""
    plan = CodemodPlanSequence.compose(
        (
            _plan(path, prefix + "second", prefix + "first"),
            CodemodPlanSequence.from_operations(
                (
                    ReplaceDeclaredCallOperation(
                        target=SourceRewriteTarget(
                            file_path=path.as_posix(), qualname=prefix + "run"
                        ),
                        callee=SourceRewriteTarget(
                            file_path=path.as_posix(), qualname=prefix + "first"
                        ),
                        expression_source="value + 1",
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                )
            ),
        )
    )
    before = subprocess.check_output([sys.executable, str(path)], text=True)
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
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["applied"]
    assert "second = first" in path.read_text(encoding="utf-8")
    assert "return (value + 1)" in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize(
    "decorator,parameters",
    (
        ("", "self, value"),
        ("@classmethod\n    ", "cls, value"),
        ("@staticmethod\n    ", "value"),
    ),
)
def test_alias_then_argument_edit_preserves_runtime_descriptor_binding(
    tmp_path: Path, decorator: str, parameters: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        f"    {decorator}def first({parameters}): return value + 1\n"
        f"    {decorator}def second({parameters}): return value + 1\n"
        "    def run(self): return self.second(3)\n"
        "print(Owner().run())\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.compose(
        (
            _plan(path, "Owner.second", "Owner.first"),
            CodemodPlanSequence.from_operations(
                (
                    ReplaceDeclaredCallArgumentsOperation(
                        target=SourceRewriteTarget(
                            file_path=path.as_posix(), qualname="Owner.run"
                        ),
                        callee=SourceRewriteTarget(
                            file_path=path.as_posix(), qualname="Owner.first"
                        ),
                        arguments_source="value=3",
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                )
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert "self.second(value=3)" in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


def _plan(
    path: Path,
    target: str = "Owner.visit_ImportFrom",
    implementation: str = "Owner.visit_Import",
) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            AliasFunctionOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname=target),
                implementation=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname=implementation
                ),
            ),
        )
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_cli_aliases_visitor_handler_without_changing_results(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "import ast\n"
        "class Owner(ast.NodeVisitor):\n"
        "    def __init__(self): self.names = []\n"
        "    def visit_Import(self, node):\n"
        "        self.names.extend(alias.name for alias in node.names)\n"
        "    def visit_ImportFrom(self, node):\n"
        "        self.names.extend(alias.name for alias in node.names)\n"
        "    sibling = 'café'\n"
        "visitor = Owner()\nvisitor.visit(ast.parse('import math; from os import path'))\n"
        "print(visitor.names, ascii(visitor.sibling))\n"
    ).replace("\n", newline)
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output(
        [sys.executable, str(path)], text=True, encoding="utf-8"
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
        input=json.dumps(json_report_object(_plan(path))),
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["applied"]
    rewritten = path.read_bytes().decode()
    assert f"    visit_ImportFrom = visit_Import{newline}    sibling" in rewritten
    assert "def visit_ImportFrom" not in rewritten
    assert (
        subprocess.check_output(
            [sys.executable, str(path)], text=True, encoding="utf-8"
        )
        == before
    )


@pytest.mark.parametrize(
    "source,target,implementation,error",
    (
        (
            "def first(): return 1\ndef second(): return 1\n",
            "first",
            "second",
            "preceding",
        ),
        (
            "def first(): return 1\nfirst = replacement\ndef second(): return 1\n",
            "second",
            "first",
            "preceding",
        ),
        (
            "def first(): return 1\nfrom external import *\ndef second(): return 1\n",
            "second",
            "first",
            "preceding",
        ),
        (
            "class A:\n def first(self): return 1\nclass B:\n def second(self): return 1\n",
            "B.second",
            "A.first",
            "lexical scope",
        ),
        (
            "def first(): return 1\ndef second():\n # preserve explanation\n return 1\n",
            "second",
            "first",
            "comments",
        ),
        ("def first(): return 1\n", "first", "first", "preceding"),
    ),
)
def test_alias_requires_available_implementation_without_discarding_comments(
    tmp_path: Path, source: str, target: str, implementation: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(source, newline="", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _plan(path, target, implementation).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


def test_alias_at_eof_preserves_absence_of_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "def first(): return 1\ndef second(): return 1", newline="", encoding="utf-8"
    )
    simulation = _plan(path, "second", "first").simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    simulation.apply()
    assert path.read_bytes().endswith(b"second = first")


def test_alias_owns_parenthesized_multiline_decorator_marker(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        "    @staticmethod\n"
        "    def first(value): return value\n"
        "    @(\n"
        "        staticmethod\n"
        "    )\n"
        "    def second(value): return value\n"
        "print(Owner.second(3))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    simulation = _plan(path, "Owner.second", "Owner.first").simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    simulation.apply()
    assert "    second = first\n" in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize(
    "decorator,parameters",
    (
        ("classmethod", "cls, value"),
        ("staticmethod", "value"),
    ),
)
def test_alias_retains_implementation_descriptor_binding(
    tmp_path: Path, decorator: str, parameters: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        f"    @{decorator}\n    def first({parameters}): return value + 1\n"
        f"    @{decorator}\n    def second({parameters}): return value + 1\n"
        "print(Owner.second(2))\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    simulation = _plan(path, "Owner.second", "Owner.first").simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


def test_documented_import_visitor_batch_runs_as_a_single_plan(tmp_path: Path) -> None:
    plan = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/import_visitor_refactor.py")
    )["PLAN"]
    path = tmp_path / "nominal_refactor_advisor/product_flow.py"
    path.parent.mkdir(parents=True)
    source = (
        "import ast\n"
        "class _DeclarationCollector(ast.NodeVisitor):\n"
        "    def __init__(self): self.names = []\n"
        "    def visit_Import(self, node: ast.Import):\n"
        "        self.names.extend(alias.name for alias in node.names)\n"
        "    def visit_ImportFrom(self, node: ast.ImportFrom):\n"
        "        self.names.extend(alias.name for alias in node.names)\n"
        "collector = _DeclarationCollector()\n"
        "collector.visit(ast.parse('import math; from os import path'))\n"
        "print(collector.names)\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    rewritten = path.read_text(encoding="utf-8")
    assert "node: ast.Import | ast.ImportFrom" in rewritten
    assert "visit_ImportFrom = visit_Import" in rewritten
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
