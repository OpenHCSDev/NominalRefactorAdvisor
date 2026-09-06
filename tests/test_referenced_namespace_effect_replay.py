"""Replay the effect factor from a reconstructed, explicitly unfactored source."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_geometry import read_source_text


@pytest.mark.parametrize("source_newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_reference_effect_factor_replays_without_replacing_leaf_bodies(
    tmp_path: Path,
    source_newline: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    relative = Path("nominal_refactor_advisor/class_namespace.py")
    expected = (
        source_newline.join(read_source_text(root / relative).splitlines())
        + source_newline
    )
    base = next(
        node
        for node in ast.parse(expected).body
        if isinstance(node, ast.ClassDef)
        and node.name == "ReferencedClassNamespaceEffect"
    )
    # Reconstruct in one spelling, then exercise the requested physical newline.
    lines = expected.splitlines()
    del lines[base.decorator_list[0].lineno - 1 : base.end_lineno]
    before = ("\n".join(lines) + "\n").replace(
        "from .descriptor_algebra import AliasProperty\n", ""
    )
    before = before.replace(
        "class NativeClassNamespaceEffect(ReferencedClassNamespaceEffect, ABC):\n",
        "@dataclass(frozen=True)\n"
        "class NativeClassNamespaceEffect(ClassNamespaceEffect, ABC):\n"
        "    node: ast.expr\n    reference: ScopedNativeReference\n\n",
    ).replace(
        "class SubscriptionClassNamespaceEffect(ReferencedClassNamespaceEffect):\n",
        "class SubscriptionClassNamespaceEffect(ClassNamespaceEffect):\n"
        "    reference: ScopedNativeReference\n",
    )
    for owner in (
        "SubscriptionClassNamespaceEffect",
        "DescriptorCallClassNamespaceEffect",
    ):
        node = next(
            node
            for node in ast.parse(before).body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
        lines = before.splitlines(keepends=True)
        lines.insert(
            node.end_lineno,
            "\n    @property\n    def recording_node(self) -> ast.AST:\n        return self.reference.node\n",
        )
        before = "".join(lines)

    before = before.replace("\n", source_newline)
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_text(before, encoding="utf-8", newline="")
    probe = """
import ast, dataclasses, importlib.util, json, sys
name = 'nominal_refactor_advisor._effect_replay'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
owner = ast.parse('class Owner:\\n    item = staticmethod(lambda: None)\\n    other: list[str] = 1\\n').body[0]
evidence = module.ClassNamespaceExecutionEvidence.from_class(owner)
print(json.dumps([
    (type(effect).__name__, ast.dump(effect.recording_node),
     [field.name for field in dataclasses.fields(effect)])
    for effect in evidence.effects
]))
"""
    original_behavior = subprocess.check_output(
        [sys.executable, "-c", probe, str(path)]
    )
    plan = runpy.run_path(
        str(root / "docs/examples/referenced_namespace_effect_refactor.py")
    )["PLAN"]
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--scan-budget-seconds",
            "0",
            "--codemod-plan",
            "-",
            "--codemod-simulate",
            "--json",
        ],
        input=json.dumps(json_report_object(plan)),
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(cli.stdout)
    assert report["plan_sequence_simulation"]["is_clean"]
    assert report["applied"] is False
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    result = plan.simulate(snapshot)
    assert result.is_clean
    assert result.stage_count == 9
    assert read_source_text(path) == before
    result.apply()
    assert ast.dump(ast.parse(path.read_text(encoding="utf-8"))) == ast.dump(
        ast.parse(expected)
    )
    assert (
        subprocess.check_output([sys.executable, "-c", probe, str(path)])
        == original_behavior
    )
