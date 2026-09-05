"""Authored call replacements retain declaration evidence and exact surroundings."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteFunctionAssignmentsOperation,
    RemoveImportNamesOperation,
    ReplaceDeclaredCallOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _operation(
    path: Path,
    expression: str,
    *,
    callee_path: Path | None = None,
    callee: str = "render",
) -> ReplaceDeclaredCallOperation:
    return ReplaceDeclaredCallOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="run"),
        callee=SourceRewriteTarget(
            file_path=(callee_path or path).as_posix(), qualname=callee
        ),
        expression_source=expression,
        selection_count=SelectionCountExpectation(exact=1),
    )


@pytest.mark.parametrize(
    "expression",
    (
        "value + 1",
        "(\n    value +\n    1\n)",
        "value + 1 # authored explanation",
        "len('''a\nb\n''')",
    ),
)
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_call_replacement_preserves_precedence_and_literal_values(
    tmp_path: Path, expression: str, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "def render(value): return value + 1\n"
        "def other(value): return value - 1\n"
        "def run(value=3):\n"
        "    café = 'untouched'\n"
        "    return café, render(value) * 2, other(value), café # retain\n"
        "print(run())\n"
    ).replace("\n", newline)
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    operation = _operation(path, expression.replace("\n", newline))
    sequence = CodemodPlanSequence.from_operations((operation,))
    replayed = CodemodPlanSequence.from_json_value(
        json.loads(json.dumps(json_report_object(sequence)))
    )
    simulation = replayed.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    assert path.read_bytes() == source.encode()
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    rewritten = path.read_bytes().decode()
    assert "other(value), café # retain" in rewritten
    assert "café = 'untouched'" in rewritten


@pytest.mark.parametrize(
    "call_source,error",
    (
        ("render(1) + render(2)", "expected exactly"),
        ("render(\n        1, # preserve\n    )", "remove a comment"),
        ("other(1)", "No resolved calls"),
    ),
)
def test_call_replacement_rejects_unproved_selection_without_writes(
    tmp_path: Path, call_source: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    source = f"def render(value): return value\ndef other(value): return value\ndef run(): return {call_source}\n"
    path.write_text(source, newline="", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        CodemodPlanSequence.from_operations((_operation(path, "42"),)).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_text(encoding="utf-8") == source


def test_saved_call_replacement_rejects_shadowed_declaration(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    source = "def render(value): return value\ndef run(): return render(1)\n"
    path.write_text(source, newline="", encoding="utf-8")
    sequence = CodemodPlanSequence.from_operations((_operation(path, "1"),))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert sequence.simulate(snapshot).is_clean
    with pytest.raises(ValueError, match="resolved calls|unresolved"):
        sequence.simulate(
            snapshot.with_virtual_sources(
                {path.as_posix(): source.replace("def run():", "def run(render):")}
            )
        )


@pytest.mark.parametrize("expression", ("1; other()", "return 1", "1) + other(2"))
def test_replacement_requires_one_expression(tmp_path: Path, expression: str) -> None:
    path = tmp_path / "probe.py"
    source = "def render(value): return value\ndef run(): return render(1)\n"
    path.write_text(source, newline="", encoding="utf-8")
    with pytest.raises(SyntaxError):
        CodemodPlanSequence.from_operations((_operation(path, expression),)).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_text(encoding="utf-8") == source


def test_cli_replaces_inherited_import_alias_without_touching_same_named_method(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library.py"
    library.write_text(
        "class Base:\n    @classmethod\n    def render(cls, value): return value + 1\n",
        newline="",
        encoding="utf-8",
    )
    path = tmp_path / "probe.py"
    source = (
        "from library import Base as Parent\n"
        "class Child(Parent): pass\n"
        "class Other:\n    @staticmethod\n    def render(value): return value - 1\n"
        "def run(value=3): return Child.render(value), Other.render(value)\n"
        "print(run())\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    sequence = CodemodPlanSequence.from_operations(
        (_operation(path, "value + 1", callee_path=library, callee="Base.render"),)
    )
    assert sequence.referenced_source_targets() == (
        sequence.documents[0].recipes[0].operations[0].target,
        sequence.documents[0].recipes[0].operations[0].callee,
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
        input=json.dumps(json_report_object(sequence)),
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    report = json.loads(cli.stdout)
    assert report["applied"]
    assert {entry["file_path"] for entry in report["base_revisions"]} >= {
        path.as_posix(),
        library.as_posix(),
    }
    assert "Child.render(value)" not in path.read_text(encoding="utf-8")
    assert "Other.render(value)" in path.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


def test_nested_selected_calls_require_separate_stages(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    source = "def render(value): return value\ndef run(): return render(render(1))\n"
    path.write_text(source, newline="", encoding="utf-8")
    operation = ReplaceDeclaredCallOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="run"),
        callee=SourceRewriteTarget(file_path=str(path), qualname="render"),
        expression_source="1",
        selection_count=SelectionCountExpectation(exact=2),
    )
    with pytest.raises(ValueError, match="[Oo]verlap"):
        CodemodPlanSequence.from_operations((operation,)).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_text(encoding="utf-8") == source


def test_historical_member_insertion_wrapper_collapses_as_one_dsl_batch(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    path = tmp_path / "probe.py"
    source = (
        "import ast\n"
        "from nominal_refactor_advisor.codemod_declaration_source import ClassBodySourceAuthority\n"
        "from nominal_refactor_advisor.codemod_source_edits import SourceTextSpanReplacement\n"
        "def run(node, source, member_sources):\n"
        "    insertion_point = ClassBodySourceAuthority(node=node, source=source)\n"
        "    insertion_offset = insertion_point.before_first_method_offset\n"
        "    return SourceTextSpanReplacement.from_offsets(\n"
        "        start_offset=insertion_offset,\n"
        "        end_offset=insertion_offset,\n"
        "        replacement_source=insertion_point.member_source(member_sources),\n"
        "    )\n"
        "source = 'class Base:\\n    pass\\n'\n"
        "print(run(ast.parse(source).body[0], source, ('    moved = 1\\n',)))\n"
    )
    path.write_text(source, newline="", encoding="utf-8")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    target = SourceRewriteTarget(file_path=str(path), qualname="run")
    operation = _operation(
        path,
        "insertion_point.member_insertion_replacement(member_sources)",
        callee_path=root / "nominal_refactor_advisor/codemod_source_edits.py",
        callee="SourceTextSpanReplacement.from_offsets",
    )
    sequence = CodemodPlanSequence.from_operations(
        (
            operation,
            DeleteFunctionAssignmentsOperation(
                target=target, assignment_names=("insertion_offset",)
            ),
            RemoveImportNamesOperation(
                target=SourceRewriteTarget(file_path=str(path)),
                module_name="nominal_refactor_advisor.codemod_source_edits",
                import_names=("SourceTextSpanReplacement",),
            ),
        )
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        (
            *parse_python_modules(root / "nominal_refactor_advisor"),
            *parse_python_modules(tmp_path),
        )
    )
    simulation = sequence.simulate(snapshot)
    assert simulation.is_clean
    assert simulation.stage_count == 3
    assert simulation.simulation.changed_file_paths == (path.as_posix(),)
    simulation.apply()
    rewritten = path.read_text(encoding="utf-8")
    assert "SourceTextSpanReplacement" not in rewritten
    assert "insertion_offset" not in rewritten
    assert "member_insertion_replacement(member_sources)" in rewritten
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
