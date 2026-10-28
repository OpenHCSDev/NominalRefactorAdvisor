"""Explicit assignment deletion preserves neighbouring Python statements."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanDocument, CodemodSourceSnapshot


def _document(
    path: Path, scope: str, target: str | None, names: tuple[str, ...]
) -> CodemodPlanDocument:
    return CodemodPlanDocument.from_payload_fields(
        {
            "recipes": [
                {
                    "recipe_id": "delete",
                    "operations": [
                        {
                            "operation": f"delete_{scope}_assignments",
                            "file_path": str(path),
                            "target_qualname": target,
                            "assignment_names": list(names),
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "scope,target,source,expression,names",
    (
        (
            "class",
            "Holder",
            "class Holder: drop = 1; keep = 2\n",
            "ns['Holder'].keep",
            ("drop",),
        ),
        (
            "class",
            "Holder",
            "class Holder: keep = 2; drop = 1\n",
            "ns['Holder'].keep",
            ("drop",),
        ),
        (
            "class",
            "Holder",
            "class Holder: drop = 1; keep = 2; drop2 = 3\n",
            "ns['Holder'].keep",
            ("drop", "drop2"),
        ),
        (
            "class",
            "Holder",
            "class Holder: drop = 1; drop2 = 3; keep = 2\n",
            "ns['Holder'].keep",
            ("drop", "drop2"),
        ),
        (
            "class",
            "Holder",
            "class Holder: keep = 2; drop = 1; drop2 = 3\n",
            "ns['Holder'].keep",
            ("drop", "drop2"),
        ),
        (
            "class",
            "Holder",
            "class Holder: drop = 1\n",
            "ns['Holder'].__name__",
            ("drop",),
        ),
        (
            "class",
            "Holder",
            "class Holder:\n    drop = 1\n",
            "ns['Holder'].__name__",
            ("drop",),
        ),
        ("module", None, "drop = 1; keep = 2\n", "ns['keep']", ("drop",)),
        ("module", None, "keep = 2; drop = 1;\n", "ns['keep']", ("drop",)),
        ("module", None, "drop = 1;\n", "'ready'", ("drop",)),
        (
            "module",
            None,
            'drop = 1; "not a docstring"\n',
            "ns.get('__doc__')",
            ("drop",),
        ),
        (
            "class",
            "Holder",
            'class Holder: drop = 1; "not a docstring"\n',
            "ns['Holder'].__doc__",
            ("drop",),
        ),
        (
            "function",
            "run",
            'def run(): drop = 1; "not a docstring"; return 2\n',
            "(ns['run'].__doc__, ns['run']())",
            ("drop",),
        ),
        (
            "function",
            "run",
            "def run(): drop = 1; return 2\n",
            "ns['run']()",
            ("drop",),
        ),
        (
            "function",
            "run",
            "def run():\n    drop = (\n        1 + 2\n    ); keep = 2\n    return keep\n",
            "ns['run']()",
            ("drop",),
        ),
        ("function", "run", "def run(): drop = 1\n", "ns['run']()", ("drop",)),
        (
            "function",
            "run",
            'def run(): "doc"; drop = 1\n',
            "(ns['run'].__doc__, ns['run']())",
            ("drop",),
        ),
        (
            "function",
            "run",
            "def run():\n\tdrop = 1\n\treturn 2 # retain\n",
            "ns['run']()",
            ("drop",),
        ),
        (
            "function",
            "run",
            'def run(): café = "é"; return 2\n',
            "ns['run']()",
            ("café",),
        ),
        (
            "function",
            "Holder.run",
            "class Holder:\n    @staticmethod\n    def run(): drop = 1; return 2\n",
            "ns['Holder'].run()",
            ("drop",),
        ),
    ),
)
def test_deletion_preserves_surviving_runtime(
    tmp_path: Path,
    scope: str,
    target: str | None,
    source: str,
    expression: str,
    names: tuple[str, ...],
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    command = [
        sys.executable,
        "-c",
        f"import runpy,sys; ns=runpy.run_path(sys.argv[1]); print({expression})",
        str(path),
    ]
    expected = subprocess.check_output(command, text=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _document(path, scope, target, names).simulate(snapshot)
    assert simulation.is_clean
    assert path.read_text(encoding="utf-8") == source
    simulation.apply()
    assert subprocess.check_output(command, text=True) == expected
    for name in names:
        assert name not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "assignment,error",
    (
        ("drop = 1\n    drop = 2", "ambiguous"),
        ("drop, *keep = (1, 2, 3)", "unselected names"),
        ("drop = obj.attr = 1", "non-name target"),
        ("drop, obj.attr = (1, 2)", "non-name target"),
    ),
)
def test_deletion_rejects_ambiguous_or_incomplete_assignment_selection(
    tmp_path: Path,
    assignment: str,
    error: str,
) -> None:
    path = tmp_path / "probe.py"
    source = f"def run():\n    {assignment}\n    return 2\n"
    path.write_text(source, newline="")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match=error):
        _document(path, "function", "run", ("drop",)).simulate(snapshot)
    assert path.read_text() == source


def test_explicit_deletion_removes_evaluation_without_touching_other_calls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "events = []\n"
        "def mark(value): events.append(value); return value\n"
        "def run(): drop = mark('removed'); keep = mark('retained'); return keep\n"
        "print(run(), events)\n"
    )
    path.write_text(source, newline="")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _document(path, "function", "run", ("drop",)).simulate(snapshot)
    simulation.apply()
    assert (
        subprocess.check_output([sys.executable, str(path)], text=True)
        == "retained ['retained']\n"
    )
