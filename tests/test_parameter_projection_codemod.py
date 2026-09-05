"""Authored parameter projections resolve lexical ownership before editing."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanDocument, CodemodSourceSnapshot


def _document(path: Path, *, projection: str = "witness.candidate") -> CodemodPlanDocument:
    return CodemodPlanDocument.from_payload_fields({
        "recipes": [{
            "recipe_id": "project-candidate",
            "operations": [{
                "operation": "project_function_parameter",
                "file_path": str(path),
                "target_qualname": "render",
                "parameter_name": "candidate",
                "projection_source": projection,
            }],
        }],
    })


@pytest.mark.parametrize(
    "body",
    (
        '    # candidate stays in this comment\n    return candidate, "candidate"\n',
        "    inner = lambda candidate: candidate + 1\n    return candidate, inner(2)\n",
        "    return [candidate for candidate in range(candidate)]\n",
        "    def inner():\n        return candidate\n    return inner()\n",
        "    def inner(value=candidate):\n        return value\n    return inner()\n",
        "    def inner():\n        global candidate\n        return candidate\n"
        "    return candidate, inner()\n",
        "    class Inner:\n        value = candidate\n    return Inner.value\n",
        "    return (lambda witness=candidate: witness)()\n",
        "    class Inner:\n        candidate = 2\n        value = candidate\n"
        "        outer = staticmethod(lambda: candidate)\n"
        "    return candidate, Inner.value, Inner.outer()\n",
    ),
    ids=("literal-comment", "lambda-shadow", "comprehension", "closure", "default",
         "explicit-global", "class-body", "default-root-shadow", "class-shadow-and-closure"),
)
def test_parameter_projection_preserves_runtime_and_shadowed_bindings(
    tmp_path: Path, body: str,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from dataclasses import dataclass\nimport json\n"
        "candidate = 41\n"
        "@dataclass(frozen=True)\nclass Witness:\n    candidate: int\n"
        "def render(candidate, witness):\n" + body
        + "print(json.dumps(render(7, Witness(7))))\n"
    )
    path.write_text(source)
    expected = subprocess.check_output([sys.executable, str(path)], text=True)
    modules = parse_python_modules(tmp_path)
    (module,) = modules
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    original_ast = ast.dump(snapshot.module_nodes_by_file_path[module.file_path], include_attributes=True)
    simulation = _document(path).simulate(snapshot)
    assert simulation.is_clean
    assert ast.dump(snapshot.module_nodes_by_file_path[module.file_path], include_attributes=True) == original_ast
    assert path.read_text() == source
    simulation.apply()
    assert "witness.candidate" in path.read_text()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == expected
    if '"candidate"' in body:
        assert '"candidate"' in path.read_text()
        assert "# candidate stays in this comment" in path.read_text()


@pytest.mark.parametrize(
    "body,projection,error",
    (
        ("    return (lambda witness: candidate)(None)\n", "witness.candidate", "captured"),
        ("    return [candidate for witness in range(2)]\n", "witness.candidate", "captured"),
        ("    candidate = 2\n    return candidate\n", "witness.candidate", "additional bindings"),
        ("    del candidate\n    return candidate\n", "witness.candidate", "additional bindings"),
        ("    def inner():\n        nonlocal candidate\n        candidate = 2\n"
         "    return candidate\n", "witness.candidate", "additional bindings"),
        ("    witness = None\n    return candidate\n", "witness.candidate", "additional bindings"),
        ("    try:\n        pass\n    except Exception as candidate:\n        pass\n"
         "    return candidate\n", "witness.candidate", "additional bindings"),
        ("    return candidate\n", "missing.candidate", "No parameter"),
        ("    return candidate\n", "witness()", "access path"),
        ("    return (lambda candidate: candidate)(2)\n", "witness.candidate", "no owned reads"),
    ),
)
def test_parameter_projection_rejects_capture_rebinding_and_unresolved_roots(
    tmp_path: Path, body: str, projection: str, error: str,
) -> None:
    path = tmp_path / "probe.py"
    source = "def render(candidate, witness):\n" + body
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(ValueError, match=error):
        _document(path, projection=projection).simulate(snapshot)
    assert path.read_text() == source


def test_saved_projection_plan_reproves_lexical_ownership(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text("def render(candidate, witness):\n    return candidate\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = _document(path)
    assert document.simulate(snapshot).is_clean
    drifted = snapshot.with_virtual_sources({
        str(path): "def render(candidate, witness):\n"
        "    return (lambda witness: candidate)(None)\n",
    })
    with pytest.raises(ValueError, match="captured"):
        document.simulate(drifted)
