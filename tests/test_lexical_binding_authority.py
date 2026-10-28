"""Shared scope ownership agrees with Python's native symbol table."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
import symtable
import textwrap

import pytest

from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor import ast_tools, lexical_bindings, python_module_identity
from nominal_refactor_advisor.declaration_dependencies import FunctionBindingProjection
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    InsertClassMemberOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize(
    "source",
    (
        "def method(value=(hidden := 3)): pass",
        "async def method(value=(hidden := 3)): pass",
        "@(hidden := staticmethod)\ndef method(): pass",
        "class Inner((base := object)): pass",
        "value = lambda item=(hidden := 3): item",
        "values = (item for item in (1, 2))",
        "def method():\n    internal = 3\n    return internal",
        "class Inner:\n    internal = 3",
        "try:\n    pass\nexcept Exception as error:\n    pass",
        "match value:\n    case {'first': captured, **rest}: pass",
    ),
)
def test_enclosing_bindings_match_native_python(source: str) -> None:
    module = symtable.symtable(
        "def probe():\n" + textwrap.indent(source, "    "), "<scope>", "exec"
    )
    native = next(
        child for child in module.get_children() if child.get_name() == "probe"
    )
    expected = frozenset(native.get_locals())
    assert (
        LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(ast.parse(source).body) == expected
    )


@pytest.mark.parametrize(
    "source",
    (
        "values = [item for item in (1, 2)]",
        "item = 0\nvalues = [item for item in (1, 2)]",
        "values = {item for item in (1, 2)}",
        "values = {item: item for item in (1, 2)}",
        "values = [(last := item) for item in (1, 2)]",
        "values = [lambda default=(last := item): default for item in (1, 2)]",
        "values = [lambda: (internal := item) for item in (1, 2)]",
    ),
)
def test_comprehension_ownership_matches_live_function_scope(source: str) -> None:
    # Runtime scope excludes compiler-private locals from comprehension inlining.
    namespace = {}
    exec(
        "def probe():\n"
        + textwrap.indent(source, "    ")
        + "\n    return frozenset(locals())\n",
        namespace,
    )
    assert (
        LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(ast.parse(source).body)
        == namespace["probe"]()
    )


def test_function_local_projection_uses_shared_header_bindings() -> None:
    source = "def outer():\n    def inner(value=(captured := 3)): pass\n    return captured\n"
    node = ast.parse(source).body[0]
    projection = FunctionBindingProjection.from_function(node)
    native = next(
        child
        for child in symtable.symtable(source, "<scope>", "exec").get_children()
        if child.get_name() == "outer"
    )
    assert projection.local_names == frozenset(native.get_locals())


@pytest.mark.parametrize(
    "member",
    (
        "def added(value=(hidden := 3)): pass",
        "@(hidden := staticmethod)\ndef added(): pass",
        "added = lambda value=(hidden := 3): value",
    ),
)
def test_insertion_rejects_additional_header_bindings(
    tmp_path: Path, member: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text("class Owner: pass\n", encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                source=member,
            ),
        )
    )
    with pytest.raises(ValueError, match="exactly one member name"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )


def test_insertion_rejects_a_binding_owned_by_an_existing_default(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "class Owner:\n    def method(value=(hidden := 3)): pass\n",
        encoding="utf-8",
        newline="",
    )
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                source="hidden = 4",
            ),
        )
    )
    with pytest.raises(ValueError, match="already binds members"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )


def test_insertion_accepts_comprehension_with_its_own_loop_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text("class Owner: pass\n", encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                source="values = [item for item in (1, 2)]",
            ),
        )
    )
    plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    namespace = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    assert namespace["Owner"].values == [1, 2]
    assert "item" not in vars(namespace["Owner"])


def test_public_imports_retain_the_same_declaration_identity() -> None:
    assert (
        ast_tools.LEXICAL_SCOPE_BINDING_AUTHORITY
        is lexical_bindings.LEXICAL_SCOPE_BINDING_AUTHORITY
    )
    assert (
        ast_tools.ImportBoundNameProjection
        is lexical_bindings.ImportBoundNameProjection
    )
    assert (
        ast_tools.PythonModulePathIdentity
        is python_module_identity.PythonModulePathIdentity
    )
    assert (
        lexical_bindings.PythonModulePathIdentity
        is python_module_identity.PythonModulePathIdentity
    )


def test_recorded_binding_plan_runs_through_cli_and_preserves_imports(
    tmp_path: Path,
) -> None:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8", newline="")
    (package / "ast_tools.py").write_text(
        "from __future__ import annotations\n"
        "import ast\n"
        "def python_module_name_is_importable(name): return name.isidentifier()\n"
        "class PythonModulePathIdentity:\n"
        "    def valid(self, name): return python_module_name_is_importable(name)\n"
        "class ImportBoundNameProjection:\n"
        "    def __init__(self, statement): self.statement = statement\n"
        "    def names(self): return tuple(alias.asname or alias.name for alias in self.statement.names)\n"
        "    def origin(self, identity: PythonModulePathIdentity): return identity\n"
        "class LexicalScopeBindingAuthority:\n"
        "    @staticmethod\n"
        "    def bound_names(nodes):\n"
        "        return frozenset(name for node in nodes for name in ImportBoundNameProjection(node).names())\n"
        "LEXICAL_SCOPE_BINDING_AUTHORITY = LexicalScopeBindingAuthority()\n",
        encoding="utf-8",
        newline="",
    )
    (package / "declaration_dependencies.py").write_text(
        "import ast\n"
        "from .ast_tools import ImportBoundNameProjection\n"
        "class _CurrentScopeBindingCollector(ast.NodeVisitor):\n"
        "    def __init__(self): self.bound_names = set()\n"
        "    def visit_Import(self, node): self.bound_names.update(ImportBoundNameProjection(node).names())\n",
        encoding="utf-8",
        newline="",
    )
    (package / "consumer.py").write_text(
        "from .ast_tools import LEXICAL_SCOPE_BINDING_AUTHORITY, PythonModulePathIdentity\n"
        "from .declaration_dependencies import _CurrentScopeBindingCollector as Collector\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import ast\n"
        "from nominal_refactor_advisor import ast_tools, consumer\n"
        "tree = ast.parse('import math as maths')\n"
        "collector = consumer.Collector(); collector.visit(tree)\n"
        "print(sorted(collector.bound_names), sorted(consumer.LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(tree.body)), consumer.PythonModulePathIdentity is ast_tools.PythonModulePathIdentity)\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(probe)], text=True)
    example = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/lexical_binding_authority_refactor.py"
    )
    plan = runpy.run_path(str(example))["PLAN"]
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
    assert "class ScopeBindingCollector" in (package / "lexical_bindings.py").read_text(
        encoding="utf-8"
    )
    assert "class _CurrentScopeBindingCollector" not in (
        package / "declaration_dependencies.py"
    ).read_text(encoding="utf-8")
    assert "class PythonModulePathIdentity" in (
        package / "python_module_identity.py"
    ).read_text(encoding="utf-8")
