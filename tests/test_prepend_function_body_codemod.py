"""Source-aware function insertion preserves suites and runtime docstrings."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanDocument, CodemodSourceSnapshot


@pytest.mark.parametrize(
    "source,target,access,asynchronous",
    (
        ('def run(value): EVENTS.append("old"); return value\n', "run", "ns['run']", False),
        ('def run(value): "doc"; EVENTS.append("old"); return value\n', "run", "ns['run']", False),
        ('def run(value): "doc"  # retained doc comment', "run", "ns['run']", False),
        ('def run(value):\n    "doc"', "run", "ns['run']", False),
        ('def run(value):\n    "doc"; return value\n', "run", "ns['run']", False),
        ('def run(value):\n    """long\ndoc"""; return value\n', "run", "ns['run']", False),
        ('def run(value): """long\ndoc"""; return value\n', "run", "ns['run']", False),
        ('def run(value):  # retained header\n    "doc"\n'
         '    # retained old body comment\n    EVENTS.append("old")\n    return value\n',
         "run", "ns['run']", False),
        ('class Runner:\n\t@classmethod\n\tdef run(cls, value):\n'
         '\t\t"doc"\n\t\treturn value\n', "Runner.run", "ns['Runner'].run", False),
        ('def decorate(fn):\n    EVENTS.append("decorated")\n    return fn\n'
         'def run(value):\n    @decorate\n    def inner():\n        return value\n'
         '    return inner()\n', "run", "ns['run']", False),
        ('async def run(\n    value,\n): "doc"; return value\n', "run", "ns['run']", True),
        ('def run(value):\n    # type: (int) -> int\n    return value\n',
         "run", "ns['run']", False),
    ),
    ids=("inline", "inline-doc", "inline-doc-only-eof", "block-doc-only-eof",
         "block-doc-shared-line", "multiline-doc-shared-line", "inline-multiline-doc",
         "comments", "tabs-classmethod", "nested-decorator", "async", "function-type-comment"),
)
def test_prepend_statements_preserves_existing_runtime(
    tmp_path: Path, source: str, target: str, access: str, asynchronous: bool,
) -> None:
    path = tmp_path / "probe.py"
    original = "EVENTS = []\n" + source
    path.write_text(original, newline="")
    call = "asyncio.run(fn(4))" if asynchronous else "fn(4)"
    command = [
        sys.executable, "-c",
        "import asyncio, json, runpy, sys; ns = runpy.run_path(sys.argv[1]); "
        f"fn = {access}; result = {call}; "
        "print(json.dumps([fn.__doc__, result, ns['EVENTS']]))",
        str(path),
    ]
    expected = json.loads(subprocess.check_output(command, text=True))
    expected[2].insert(0, "new\ntext")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument.from_payload_fields({
        "recipes": [{
            "recipe_id": "prepend-body",
            "operations": [{
                "operation": "prepend_function_body",
                "file_path": str(path),
                "target_qualname": target,
                "body_source": 'EVENTS.append("""new\ntext""")',
            }],
        }],
    })
    simulation = document.simulate(snapshot)
    assert simulation.is_clean
    assert path.read_text() == original
    simulation.apply()
    assert json.loads(subprocess.check_output(command, text=True)) == expected
    before_node = next(
        node for node in ast.walk(ast.parse(original, type_comments=True))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    )
    after_node = next(
        node for node in ast.walk(ast.parse(path.read_text(), type_comments=True))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    )
    assert after_node.type_comment == before_node.type_comment
    for comment in ("# retained doc comment", "# retained header", "# retained old body comment"):
        if comment in original:
            assert comment in path.read_text()
