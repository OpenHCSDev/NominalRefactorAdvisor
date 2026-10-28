"""Declaration refactors preserve Python's ordered class namespace lookup."""

import ast
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    RenameTopLevelBindingAuthorityOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyProjection,
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.lexical_bindings import LexicalScopeBindingAuthority


@pytest.mark.parametrize(
    "body",
    (
        "def method(value=(hidden := 3)): pass\ncopy = hidden",
        "async def method(value=(hidden := 3)): pass\ncopy = hidden",
        "def method(value=(hidden := 3), other=hidden): pass\ncopy = hidden",
        "value = lambda arg=(hidden := 3): arg\ncopy = hidden",
        "copy = ((hidden := 3), hidden)[1]",
        "hidden = 3\ndel hidden\ncopy = hidden",
        "if flag:\n    hidden = 3\nelse:\n    hidden = 4\ncopy = hidden",
        "if flag:\n    hidden = 3\n    copy = hidden\nelse:\n    copy = hidden",
        "copy = (hidden := 3) and hidden",
        "copy = hidden if (hidden := 3) else hidden",
        "data = {0: (hidden := 3), hidden: hidden}\ncopy = data[hidden]",
        "hidden = 3\nhidden += 1\ncopy = hidden",
        "before = hidden\nhidden: int = 3\ncopy = before + hidden",
        "class Inner((hidden := object)): pass\ncopy = hidden.__name__",
    ),
)
@pytest.mark.parametrize("flag", (False, True))
def test_rename_preserves_native_class_lookup(
    tmp_path: Path,
    body: str,
    flag: bool,
) -> None:
    source = (
        f"hidden = 99\nflag = {flag}\nclass Owner:\n"
        + textwrap.indent(body, "    ")
        + "\nprint(Owner.copy)\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=path.as_posix()),
        binding_name="hidden",
        new_name="renamed",
    )
    CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before


@pytest.mark.parametrize(
    "body",
    (
        "if flag:\n    hidden = 3\ncopy = hidden",
        "flag and (hidden := 3)\ncopy = hidden",
        "(hidden := 3) if flag else None\ncopy = hidden",
        "hidden = 3\nif flag:\n    del hidden\ncopy = hidden",
        "for item in values:\n    hidden = item\ncopy = hidden",
        "try:\n    hidden = factory()\nexcept Exception:\n    pass\ncopy = hidden",
        "try:\n    hidden = factory()\nexcept Exception:\n    copy = hidden",
        "for item in values:\n    hidden = item\n    copy = hidden",
        "hidden += 1\ncopy = hidden",
    ),
)
def test_conditional_class_binding_remains_unproved(body: str) -> None:
    projection = ModuleLexicalDependencyProjection.from_module(
        ast.parse("class Owner:\n" + textwrap.indent(body, "    "))
    )
    with pytest.raises(ValueError, match="class.*binding"):
        projection.external_references_named("hidden")


def test_unrelated_uncertainty_does_not_block_exact_reference() -> None:
    projection = ModuleLexicalDependencyProjection.from_module(
        ast.parse(
            "class Owner:\n"
            "    if flag:\n"
            "        hidden = 3\n"
            "    copy = hidden\n"
            "    other = External\n"
        )
    )
    assert len(projection.external_references_named("External")) == 1


def test_move_dependency_rejects_unproved_class_lookup() -> None:
    declaration = ast.parse(
        "class Owner:\n    if flag:\n        hidden = 3\n    copy = hidden\n"
    ).body[0]
    with pytest.raises(ValueError, match="class.*binding"):
        DeclarationDependencyProjection.from_declarations((declaration,))


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_cli_composes_renames_without_rewriting_class_owned_reads(
    tmp_path: Path,
    newline: str,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "hidden = 99\nclass Owner:\n"
        "    def method(value=(hidden := 3), other=hidden): pass\n"
        "    copy = hidden\nprint(Owner.copy, hidden)\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    plan = CodemodPlanSequence.from_operations(
        tuple(
            RenameTopLevelBindingAuthorityOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                binding_name=old,
                new_name=new,
            )
            for old, new in (("hidden", "intermediate"), ("intermediate", "renamed"))
        )
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(json_report_object(plan)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.check_output([sys.executable, str(path)]) == before
    assert b"copy = hidden" in path.read_bytes()
    assert b"print(Owner.copy, renamed)" in path.read_bytes()


def test_rename_refuses_ambiguous_lookup_without_touching_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "hidden = 99\nclass Owner:\n"
        "    if flag:\n        hidden = 3\n    copy = hidden\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            RenameTopLevelBindingAuthorityOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                binding_name="hidden",
                new_name="renamed",
            ),
        )
    )
    with pytest.raises(ValueError, match="class.*binding"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode("utf-8")


def test_class_global_directive_does_not_override_a_methods_closure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "hidden = 99\n"
        "def outer():\n"
        "    hidden = 3\n"
        "    class Owner:\n"
        "        global hidden\n"
        "        def method(self):\n"
        "            return hidden\n"
        "    return Owner\n"
        "print(outer()().method(), hidden)\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(path)])
    assert before.strip() == b"3 99"
    projection = ModuleLexicalDependencyProjection.from_module(
        ast.parse(path.read_text(encoding="utf-8"))
    )
    assert tuple(
        reference.lineno for reference in projection.external_references_named("hidden")
    ) == (9,)


def test_scope_refactor_example_moves_consumers_and_preserves_identity(
    tmp_path: Path,
) -> None:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "declaration_dependencies.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "def _argument_names(values):\n    return set(values)\n"
        "@dataclass(frozen=True)\n"
        "class FunctionBindingProjection:\n"
        "    names: frozenset[str]\n"
        "    @classmethod\n"
        "    def from_values(cls, values):\n"
        "        return cls(frozenset(_argument_names(values)))\n",
        encoding="utf-8",
        newline="",
    )
    (package / "lexical_scopes.py").write_text(
        "from __future__ import annotations\nclass ExistingScope: pass\n",
        encoding="utf-8",
        newline="",
    )
    consumer = package / "consumer.py"
    consumer.write_text(
        "from .declaration_dependencies import FunctionBindingProjection\n"
        "value = FunctionBindingProjection.from_values(('alpha', 'beta', 'alpha'))\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from nominal_refactor_advisor.consumer import value\n"
        "print(sorted(value.names))\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(probe)])
    example = (
        Path(__file__).resolve().parents[1] / "docs/examples/lexical_scope_refactor.py"
    )
    plan = runpy.run_path(str(example))["PLAN"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(json_report_object(plan)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(package),
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "from .lexical_scopes import FunctionBindingProjection"
        in consumer.read_text(encoding="utf-8")
    )
    assert subprocess.check_output([sys.executable, str(probe)]) == before
    probe.write_text(
        "from nominal_refactor_advisor import declaration_dependencies, lexical_scopes\n"
        "assert declaration_dependencies.FunctionBindingProjection is "
        "lexical_scopes.FunctionBindingProjection\n",
        encoding="utf-8",
        newline="",
    )
    subprocess.run([sys.executable, str(probe)], check=True)


@pytest.mark.parametrize(
    "body, expected",
    (
        ("copy = hidden\nhidden = 4", b"99"),
        ("hidden = 4\ndel hidden\ncopy = hidden", b"99"),
        ("hidden: int\ncopy = hidden", b"99"),
        ("copy = hidden", b"3"),
    ),
)
def test_class_declarations_distinguish_module_fallback_from_closure_lookup(
    tmp_path: Path,
    body: str,
    expected: bytes,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "hidden = 99\ndef outer():\n    hidden = 3\n    class Owner:\n"
        + textwrap.indent(body, "        ")
        + "\n    return Owner\nprint(outer().copy)\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    assert before.strip() == expected
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=path.as_posix()),
        binding_name="hidden",
        new_name="renamed",
    )
    CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_generic_method_annotation_retains_class_namespace_lookup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        "from typing import get_type_hints\n"
        "hidden = str\nclass Owner:\n    hidden = int\n"
        "    def method[T](self, value: hidden): pass\n"
        "print(get_type_hints(Owner.method)['value'].__name__, hidden.__name__)\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(path)])
    assert before.strip() == b"int str"
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=path.as_posix()),
        binding_name="hidden",
        new_name="renamed",
    )
    CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert subprocess.check_output([sys.executable, str(path)]) == before


def test_argument_binding_example_replaces_calls_and_removes_duplicate_helper(
    tmp_path: Path,
) -> None:
    package = tmp_path / "nominal_refactor_advisor"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "lexical_bindings.py").write_text(
        "from __future__ import annotations\nimport ast\n"
        "class LexicalScopeBindingAuthority:\n"
        + textwrap.indent(
            textwrap.dedent(
                inspect.getsource(LexicalScopeBindingAuthority.argument_names)
            ),
            "    ",
        )
        + "\nLEXICAL_SCOPE_BINDING_AUTHORITY = LexicalScopeBindingAuthority()\n",
        encoding="utf-8",
        newline="",
    )
    scopes = package / "lexical_scopes.py"
    scopes.write_text(
        "def _argument_names(arguments):\n"
        "    return {argument.arg for argument in (\n"
        "        *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,\n"
        "        arguments.vararg, arguments.kwarg) if argument is not None}\n"
        "class FunctionBindingProjection:\n"
        "    @classmethod\n"
        "    def from_function(cls, node):\n"
        "        return frozenset(_argument_names(node.args))\n",
        encoding="utf-8",
        newline="",
    )
    dependencies = package / "declaration_dependencies.py"
    dependencies.write_text(
        "from .lexical_scopes import _argument_names\n"
        "class FunctionParameterBinding:\n"
        "    def __init__(self, node):\n        self.node = node\n"
        "    def without_binding(self):\n"
        "        return frozenset(_argument_names(self.node.args))\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import ast\n"
        "from nominal_refactor_advisor.lexical_scopes import FunctionBindingProjection\n"
        "from nominal_refactor_advisor.declaration_dependencies import FunctionParameterBinding\n"
        "node = ast.parse('def f(pos, /, normal, *args, keyword, **kwargs): pass').body[0]\n"
        "left = FunctionBindingProjection.from_function(node)\n"
        "right = FunctionParameterBinding(node).without_binding()\n"
        "assert left == right\nprint(sorted(left))\n",
        encoding="utf-8",
        newline="",
    )
    before = subprocess.check_output([sys.executable, str(probe)])
    example = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/argument_binding_refactor.py"
    )
    plan = runpy.run_path(str(example))["PLAN"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(json_report_object(plan)), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(package),
            "--codemod-plan",
            str(plan_path),
            "--codemod-apply",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.check_output([sys.executable, str(probe)]) == before
    assert "_argument_names" not in scopes.read_text(encoding="utf-8")
    assert "_argument_names" not in dependencies.read_text(encoding="utf-8")
