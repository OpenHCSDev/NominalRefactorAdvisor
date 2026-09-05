"""A recorded multi-stage extraction runs through the CLI and retains identities."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor import call_binding, product_flow, value_expression
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_documented_extraction_can_split_its_new_module_in_the_next_stage(
    tmp_path: Path, newline: str
) -> None:
    plan = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/call_binding_extraction.py")
    )["PLAN"]
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8", newline="")
    source = (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "class CompactValueExpression:\n"
        "    @staticmethod\n"
        "    def project(value):\n"
        "        return LexicalValueReference(value) if value else OpaqueValueExpression()\n"
        "@dataclass(frozen=True)\n"
        "class LexicalValueReference(CompactValueExpression):\n"
        "    value: str\n"
        "class OpaqueValueExpression(CompactValueExpression): pass\n"
        "class CompactFunctionSignature:\n"
        "    def bind(self, value): return CompactValueExpression.project(value)\n"
        "class Unrelated: pass\n"
    ).replace("\n", newline)
    (package / "product_flow.py").write_text(source, encoding="utf-8", newline="")
    (package / "consumer.py").write_text(
        "from .product_flow import CompactFunctionSignature, LexicalValueReference, Unrelated\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from nominal_refactor_advisor import product_flow, consumer\n"
        "result = consumer.CompactFunctionSignature().bind('sample')\n"
        "print(result.value, type(result) is product_flow.LexicalValueReference, "
        "consumer.Unrelated is product_flow.Unrelated)\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(probe)], text=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(package),
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
    assert subprocess.check_output([sys.executable, str(probe)], text=True) == before
    declarations = {
        path.name: {
            node.name
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)
        }
        for path in package.glob("*.py")
    }
    assert declarations["product_flow.py"] == {"Unrelated"}
    assert declarations["call_binding.py"] == {"CompactFunctionSignature"}
    assert declarations["value_expression.py"] == {
        "CompactValueExpression",
        "LexicalValueReference",
        "OpaqueValueExpression",
    }
    consumer = (package / "consumer.py").read_text(encoding="utf-8")
    assert "from .call_binding import CompactFunctionSignature" in consumer
    assert "from .value_expression import LexicalValueReference" in consumer


@pytest.mark.parametrize("owner", (call_binding, value_expression))
def test_extracted_public_declarations_retain_one_runtime_identity(owner) -> None:
    declarations = tuple(
        value
        for value in vars(owner).values()
        if isinstance(value, type) and value.__module__ == owner.__name__
    )
    assert declarations
    for declaration in declarations:
        assert vars(product_flow)[declaration.__name__] is declaration


def test_value_expression_and_binding_owners_do_not_import_product_flow() -> None:
    for owner in (value_expression, call_binding):
        module = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
        imports = tuple(
            node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
        )
        assert all(node.module != "product_flow" for node in imports)
    assert call_binding.LexicalValueReference is value_expression.LexicalValueReference
