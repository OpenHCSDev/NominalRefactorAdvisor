"""One lexical frame stack supplies lookup, provenance and exception cleanup."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleBindingResolutionPhase,
    _DeclarationDependencyCollector,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.lexical_scopes import (
    ClassNamespaceScope,
    FunctionBindingProjection,
    LexicalScopeContext,
)


def test_class_provenance_is_derived_from_the_same_frames_as_lookup() -> None:
    outer, inner = ast.parse("class Outer: pass\nclass Inner: pass\n").body
    function = ast.parse("def method(value): pass").body[0]
    context = LexicalScopeContext()
    outer_frame = ClassNamespaceScope(outer)
    assert outer_frame.declarations is outer_frame.declarations
    with context._scope(outer_frame):
        assert context.owner_classes == (outer,)
        with context._scope(FunctionBindingProjection.from_function(function)):
            assert context.owner_classes == (outer,)
            assert context._active_class_scope is None
            with context._scope(ClassNamespaceScope(inner)):
                assert context.owner_classes == (outer, inner)
        assert context.owner_classes == (outer,)
    assert context.scopes == []
    assert context.owner_classes == ()
    with pytest.raises(AttributeError):
        context.owner_classes = ()
    assert LexicalScopeContext() != LexicalScopeContext()


class StopTraversal(Exception):
    pass


class InterruptingCollector(_DeclarationDependencyCollector):
    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "interrupt":
            raise StopTraversal
        super().visit_Name(node)


@pytest.mark.parametrize(
    "source",
    (
        "class Owner:\n    value = interrupt",
        "class Owner:\n    def method(self): return interrupt",
        "class Owner:\n    async def method(self): return interrupt",
        "def outer(): return lambda: interrupt",
        "class Owner:\n    values = [interrupt for item in (1, 2)]",
        "class Owner:\n    values = {interrupt for item in (1, 2)}",
        "class Owner:\n    values = {item: interrupt for item in (1, 2)}",
        "class Owner:\n    values = (interrupt for item in (1, 2))",
    ),
)
def test_interrupted_traversal_unwinds_all_lexical_frames(source: str) -> None:
    collector = InterruptingCollector()
    with pytest.raises(StopTraversal):
        collector.visit(ast.parse(source))
    assert collector.scopes == []
    assert collector.owner_classes == ()
    assert collector.use is DeclarationDependencyUse.EXECUTION
    assert collector.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
    collector.visit(ast.parse("class Next: pass"))
    assert collector.owner_classes == ()


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_interrupted_type_parameter_scope_unwinds() -> None:
    collector = InterruptingCollector()
    with pytest.raises(StopTraversal):
        collector.visit(
            ast.parse("class Owner:\n    def method[T: interrupt](self): pass")
        )
    assert collector.scopes == []
    assert collector.owner_classes == ()


def test_cli_composes_member_promotion_and_module_move(tmp_path: Path) -> None:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "declaration_dependencies.py").write_text(
        "from __future__ import annotations\n"
        "class LexicalScopeContext:\n    scopes = ('owned',)\n"
        "class _DeclarationDependencyCollector:\n"
        "    scopes = ('owned',)\n"
        "    def _resolve_name(self, name):\n        return name in self.scopes\n"
        "    @property\n"
        "    def _active_class_scope(self):\n        return self.scopes[-1]\n",
        encoding="utf-8",
        newline="",
    )
    (package / "lexical_scopes.py").write_text(
        "from __future__ import annotations\nclass Existing: pass\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from nominal_refactor_advisor.declaration_dependencies import _DeclarationDependencyCollector\n"
        "collector = _DeclarationDependencyCollector()\n"
        "print(collector._resolve_name('owned'), collector._active_class_scope)\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(probe)])
    example = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/lexical_scope_context_refactor.py"
    )
    plan = runpy.run_path(str(example))["OWNERSHIP_PLAN"]
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
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    assert subprocess.check_output([sys.executable, str(probe)]) == before
    probe.write_text(
        "from nominal_refactor_advisor import declaration_dependencies, lexical_scopes\n"
        "owner = lexical_scopes.LexicalScopeContext\n"
        "consumer = declaration_dependencies._DeclarationDependencyCollector\n"
        "assert owner is declaration_dependencies.LexicalScopeContext\n"
        "assert consumer._resolve_name is owner._resolve_name\n"
        "assert '_resolve_name' not in vars(consumer)\n",
        encoding="utf-8",
        newline="",
    )
    subprocess.run([sys.executable, str(probe)], check=True)
