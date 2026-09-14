"""Replay a historical field factor on a standalone, fixed source specimen.

This tests DSL geometry and behaviour preservation. The removed reference
carriers are not current native-admission authorities.
"""

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

UNFACTORED_SOURCE = """\
import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopedNativeReference:
    node: ast.expr
    resolution: str


@dataclass(frozen=True)
class ClassNamespaceEffect(ABC):
    node: ast.AST
    use: str

    @abstractmethod
    def observation(self):
        raise NotImplementedError


@dataclass(frozen=True)
class NativeClassNamespaceEffect(ClassNamespaceEffect, ABC):
    node: ast.expr
    reference: ScopedNativeReference

    def reference_observation(self):
        return ast.dump(self.reference.node), self.reference.resolution


class DescriptorClassNamespaceEffect(NativeClassNamespaceEffect):
    def observation(self):
        return self.use, self.reference_observation()


@dataclass(frozen=True)
class SubscriptionClassNamespaceEffect(ClassNamespaceEffect):
    node: ast.Subscript
    reference: ScopedNativeReference

    def observation(self):
        return self.use, ast.dump(self.node.slice), self.reference.resolution
"""


@pytest.mark.parametrize("source_newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_reference_effect_factor_replays_without_replacing_leaf_bodies(
    tmp_path: Path,
    source_newline: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    relative = Path("nominal_refactor_advisor/class_namespace.py")
    before = UNFACTORED_SOURCE.replace("\n", source_newline)
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_text(before, encoding="utf-8", newline="")
    probe = """
import ast, dataclasses, importlib.util, inspect, json, sys
name = '_historical_effect_replay'
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
descriptor = ast.parse('property', mode='eval').body
subscription = ast.parse('list[str]', mode='eval').body
effects = (
    module.DescriptorClassNamespaceEffect(
        descriptor, 'decorator', module.ScopedNativeReference(descriptor, 'external')
    ),
    module.SubscriptionClassNamespaceEffect(
        subscription, 'annotation', module.ScopedNativeReference(subscription.value, 'external')
    ),
)
print(json.dumps([
    (type(effect).__name__, effect.observation(),
     [field.name for field in dataclasses.fields(effect)],
     str(inspect.signature(type(effect))))
    for effect in effects
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
    assert result.stage_count == 6
    assert read_source_text(path) == before
    result.apply()
    after = read_source_text(path)
    original_classes = {
        node.name: node
        for node in ast.parse(before).body
        if isinstance(node, ast.ClassDef)
    }
    factored_classes = {
        node.name: node
        for node in ast.parse(after).body
        if isinstance(node, ast.ClassDef)
    }
    shared = factored_classes["ReferencedClassNamespaceEffect"]
    assert [
        node.target.id for node in shared.body if isinstance(node, ast.AnnAssign)
    ] == [
        "node",
        "reference",
    ]
    for name, original in original_classes.items():
        factored = factored_classes[name]
        original_methods = [
            ast.dump(node)
            for node in original.body
            if isinstance(node, ast.FunctionDef)
        ]
        factored_methods = [
            ast.dump(node)
            for node in factored.body
            if isinstance(node, ast.FunctionDef)
        ]
        assert factored_methods == original_methods
    for name in ("NativeClassNamespaceEffect", "SubscriptionClassNamespaceEffect"):
        factored = factored_classes[name]
        assert ast.unparse(factored.bases[0]) == shared.name
        assert all(
            not isinstance(node, ast.AnnAssign) or node.target.id != "reference"
            for node in factored.body
        )
    assert (
        subprocess.check_output([sys.executable, "-c", probe, str(path)])
        == original_behavior
    )
