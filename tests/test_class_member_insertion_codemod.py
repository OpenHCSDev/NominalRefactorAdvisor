"""Authored class members reuse collision, coalescence and source geometry."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    InsertClassMemberOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_declaration_source import ClassMemberInsertion
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("indentation", ("    ", "\t", "        "))
def test_cli_derives_member_indentation_and_preserves_decorators_and_literals(
    tmp_path: Path, newline: str, indentation: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Outer:\n"
        f"{indentation}class Owner:\n"
        f"{indentation * 2}# existing method\n"
        f"{indentation * 2}@(\n"
        f"{indentation * 2}    staticmethod\n"
        f"{indentation * 2})\n"
        f"{indentation * 2}def original(): return 3\n"
        "print(Outer.Owner.original())\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    member = "# new property\n@property\ndef text(self):\n    return '''café\n  literal indent\nlast'''\n"
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Outer.Owner"
                ),
                source=member,
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
    rewritten = path.read_bytes().decode()
    assert indentation * 2 + "def text(self):" in rewritten
    assert source[source.index(indentation * 2 + "# existing method") :] in rewritten
    namespace = {}
    exec(rewritten, namespace)
    assert namespace["Outer"].Owner().text == "café\n  literal indent\nlast"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from probe import Outer; assert Outer.Owner.original() == 3; assert len(Outer.Owner().text) == 26",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("inline", (False, True))
def test_one_document_keeps_coalesced_member_evaluation_order(
    tmp_path: Path, inline: bool
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner: pass\n"
        if inline
        else "class Owner:\n    def original(self): return 1\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    target = SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner")
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                recipe_id="dependent-members",
                operations=(
                    InsertClassMemberOperation(target=target, source="z = 3"),
                    InsertClassMemberOperation(target=target, source="a = z + 1"),
                ),
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = document.recipes[0].operations[0]
    assert isinstance(operation.source_edits(snapshot)[0], ClassMemberInsertion)
    document.simulate(snapshot).apply()
    namespace = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    assert namespace["Owner"].a == 4


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_atomic_method_alias_captures_class_member_not_same_spelled_global(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        (
            "events = []\n"
            "def visit(self): return 'global'\n"
            "class Owner:\n    def existing(self): return 'existing'\n"
        ).replace("\n", newline),
        encoding="utf-8",
        newline="",
    )
    target = SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner")
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                recipe_id="class-owned-alias",
                operations=(
                    InsertClassMemberOperation(
                        target=target,
                        source="def visit(self, value=events.append('default')): return 'member'",
                    ),
                    InsertClassMemberOperation(target=target, source="alias = visit"),
                ),
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document.simulate(snapshot).apply()
    namespace = {}
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
    owner = namespace["Owner"]
    assert owner.alias is owner.visit
    assert owner.alias is not namespace["visit"]
    assert owner().alias() == "member"
    assert namespace["events"] == ["default"]


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_separate_stages_recompute_current_first_method_boundary(
    tmp_path: Path, reverse: bool, newline: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        (
            "def visit(self): return 'global'\n"
            "class Owner:\n    def existing(self): return 'existing'\n"
        ).replace("\n", newline),
        encoding="utf-8",
        newline="",
    )
    target = SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner")
    sources = ("def visit(self): return 'member'", "alias = visit")
    if reverse:
        sources = tuple(reversed(sources))
    plan = CodemodPlanSequence.from_operations(
        InsertClassMemberOperation(target=target, source=source) for source in sources
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    plan.simulate(snapshot).apply()
    namespace = {}
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
    owner = namespace["Owner"]
    # Sequential edit order is not class-body statement order. No dependency
    # inference silently moves this authored global capture below the method.
    assert owner.alias is namespace["visit"]
    assert owner.alias is not owner.visit
    assert owner().alias() == "global"


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("inline", (False, True))
def test_documented_atomic_batch_then_body_edit_roundtrips_through_cli(
    tmp_path: Path, newline: str, inline: bool
) -> None:
    path = tmp_path / "visitor.py"
    class_source = (
        "class Visitor: 'docs'; pass\n"
        if inline
        else "class Visitor:\n    'docs'\n    def existing(self): return 'existing'\n"
    )
    path.write_text(
        ("def visit(self): return 'global'\n" + class_source).replace("\n", newline),
        encoding="utf-8",
        newline="",
    )
    example = Path(__file__).parents[1] / "docs/examples/cohesive_class_members.py"
    payload = subprocess.check_output(
        [sys.executable, str(example), str(path)], text=True
    )
    # Exercise the published emitter rather than reproducing its plan in tests.
    plan = runpy.run_path(str(example))["build_plan"](str(path))
    assert json.loads(payload) == json.loads(json.dumps(json_report_object(plan)))
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
        input=payload,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from visitor import Visitor, visit; "
            "assert Visitor.visit_alias is Visitor.visit; "
            "assert Visitor.visit_alias is not visit; "
            "assert Visitor().visit_alias() == 'updated'; "
            "assert Visitor.__doc__ == 'docs'",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "member,name",
    (
        ("field: int", "field"),
        ("field = 4", "field"),
        ("async def fetch(self): return 1", "fetch"),
        ("class Nested: value = 3", "Nested"),
    ),
)
def test_member_name_is_derived_from_authored_declaration(
    tmp_path: Path, member: str, name: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text("class Owner: 'docs'", encoding="utf-8", newline="")
    operation = InsertClassMemberOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
        source=member,
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert operation.source_edits(snapshot)[0].members[0].name == name
    CodemodPlanSequence.from_operations((operation,)).simulate(snapshot).apply()
    node = ast.parse(path.read_text(encoding="utf-8")).body[0]
    assert ast.get_docstring(node) == "docs"


@pytest.mark.parametrize(
    "member",
    (
        "existing = 3",
        "def existing(self): pass",
        "first = second = 1",
        "a = 1\nb = 2",
        "print('side effect')",
        "value = (",
        "# comment only",
        "obj.attr = 3",
        "from somewhere import value",
    ),
)
def test_invalid_or_colliding_members_leave_source_unchanged(
    tmp_path: Path, member: str
) -> None:
    path = tmp_path / "probe.py"
    source = "class Owner:\n    existing = 1\n"
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                source=member,
            ),
        )
    )
    with pytest.raises(ValueError):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


def test_member_insertion_rejects_a_function_scope(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text("def Owner(): pass\n", encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                source="value = 3",
            ),
        )
    )
    with pytest.raises(ValueError, match="class declaration"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
