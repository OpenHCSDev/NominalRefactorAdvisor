"""Authored caller migrations retain declaration identity and source boundaries."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanDocument, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry


def _document(
    path: Path, arguments: str, callee: str = "Base.render"
) -> CodemodPlanDocument:
    return CodemodPlanDocument.from_payload_fields(
        {
            "recipes": [
                {
                    "recipe_id": "caller",
                    "operations": [
                        {
                            "operation": "replace_declared_call_arguments",
                            "file_path": str(path),
                            "target_qualname": "run",
                            "callee": {
                                "file_path": str(path),
                                "target_qualname": callee,
                            },
                            "arguments_source": arguments,
                            "selection_count": {"exact": 1},
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "callee_source", ("Child.render", "(Child.render)", "((Child.render))")
)
def test_inherited_call_migration_preserves_runtime_and_other_calls(
    tmp_path: Path, callee_source: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Base:\n"
        "    @classmethod\n    def render(cls, value, *, offset=2): return value + offset\n"
        "class Child(Base): pass\n"
        "class Other:\n    @staticmethod\n    def render(value): return value * 3\n"
        "def run():\n"
        f"    return {callee_source}(\n        4,\n        offset=2,\n    ), Other.render(4)  # retain\n"
        "print(run())\n"
    )
    path.write_text(source)
    expected = subprocess.check_output([sys.executable, str(path)], text=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = _document(path, "4").simulate(snapshot)
    assert simulation.is_clean
    assert path.read_text() == source
    simulation.apply()
    assert f"{callee_source}(4), Other.render(4)  # retain" in path.read_text()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == expected


@pytest.mark.parametrize(
    "arguments,error",
    (
        ("", "missing_required_argument"),
        ("1, 2, 3", "too_many_positional_arguments"),
        ("1, value=2", "duplicate_argument"),
        ("*values", "variadic_unpacking"),
        ("**values", "variadic_unpacking"),
        ("1) + _nra_call_(2", "one call argument list"),
    ),
)
def test_replacement_rejects_unproved_argument_binding(
    tmp_path: Path, arguments: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    source = "def render(value): return value\ndef run(): return render(1)\n"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match=error):
        _document(path, arguments, "render").simulate(snapshot)
    assert path.read_text() == source


def test_saved_call_edit_rejects_shadowed_authority_and_comment_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = "def render(value): return value\ndef run(): return render(1)\n"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = _document(path, "2", "render")
    assert document.simulate(snapshot).is_clean
    for changed, error in (
        (source.replace("def run():", "def run(render):"), "resolved calls|unresolved"),
        (source.replace("render(1)", "render(\n 1, # retain\n)"), "remove a comment"),
    ):
        with pytest.raises(ValueError, match=error):
            document.simulate(snapshot.with_virtual_sources({str(path): changed}))
    assert path.read_text() == source


def test_import_alias_resolves_declaring_module_and_retains_argument_order(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library.py"
    library.write_text("def render(first, *, second): return first, second\n")
    path = tmp_path / "probe.py"
    source = (
        "from library import render as chosen\n"
        "events = []\n"
        "def mark(value):\n    events.append(value)\n    return value\n"
        "def run(): return chosen(mark(1), second=mark(2))\n"
        "print(run(), events)\n"
    )
    path.write_text(source)
    expected = subprocess.check_output([sys.executable, str(path)], text=True)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument.from_payload_fields(
        {
            "recipes": [
                {
                    "recipe_id": "imported-call",
                    "operations": [
                        {
                            "operation": "replace_declared_call_arguments",
                            "file_path": str(path),
                            "target_qualname": "run",
                            "callee": {
                                "file_path": str(library),
                                "target_qualname": "render",
                            },
                            "arguments_source": "first=mark(1), second=mark(2)",
                            "selection_count": {"exact": 1},
                        }
                    ],
                }
            ],
        }
    )
    simulation = document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert "chosen(first=mark(1), second=mark(2))" in path.read_text()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == expected


def test_argument_replacement_rejects_signature_changing_decorator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "def decorate(fn): return fn\n"
        "@decorate\ndef render(value): return value\n"
        "def run(): return render(1)\n"
    )
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match="signature_decorator_hazard"):
        _document(path, "2", "render").simulate(snapshot)
    assert path.read_text() == source


@pytest.mark.parametrize(
    "call",
    (
        "(café)(inner(3), label='()')",
        "factory()(1, (2, 3))",
        "((lambda x: x))(1)",
        "café(f'{inner(3)}', x=(1 + 2))",
    ),
)
def test_call_argument_geometry_uses_outer_call_pair(call: str) -> None:
    source = f"result = {call}\n"
    node = ast.parse(source).body[0].value
    assert isinstance(node, ast.Call)
    geometry = SourceTextGeometry(source)
    span = geometry.call_argument_span(node)
    edited = source[: span.start_offset] + "42" + source[span.end_offset :]
    expected = ast.parse(edited).body[0].value
    assert ast.dump(expected.func) == ast.dump(node.func)
    assert len(expected.args) == 1 and expected.args[0].value == 42
