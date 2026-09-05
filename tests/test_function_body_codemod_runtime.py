"""Body replacements preserve declaration ownership and literal values."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodSourceSnapshot,
)


@pytest.mark.parametrize(
    "source,target,call",
    (
        ("def run(value): return value\n", "run", "run(4)"),
        (
            "def run(value):  # retained header\n"
            "    # Old body explanation.\n"
            "    return value\n",
            "run", "run(4)",
        ),
        (
            "def run(value):\n"
            "    @old_decorator\n"
            "    def inner():\n"
            "        return value\n"
            "    return inner()\n",
            "run", "run(4)",
        ),
        (
            "class Runner:\n"
            "    @classmethod\n"
            "    def run(cls, value): return value\n",
            "Runner.run", "Runner.run(4)",
        ),
        (
            "class Runner:\n"
            "\t@staticmethod\n"
            "\tdef run(value):\n"
            "\t\treturn value\n",
            "Runner.run", "Runner.run(4)",
        ),
        (
            "async def run(\n    value,\n): return value\n",
            "run", "asyncio.run(run(4))",
        ),
    ),
    ids=("inline", "block-comments", "nested-decorator", "classmethod", "tabs", "async"),
)
@pytest.mark.parametrize(
    "body,expected",
    (
        ('return """first\nsecond""", value + 1', ["first\nsecond", 5]),
        ('return f"""first\n{value + 1}\nlast"""', "first\n5\nlast"),
        (
            'return f"""first\n{(\n    value + 1\n)}\nlast"""',
            "first\n5\nlast",
        ),
        pytest.param(
            'item = t"""first\n{(\n    value + 1\n)}\nlast"""\n'
            "return item.strings, item.interpolations[0].expression",
            [["first\n", "\nlast"], "(\n    value + 1\n)"],
            marks=pytest.mark.skipif(
                sys.version_info < (3, 14), reason="Template strings need Python 3.14",
            ),
        ),
    ),
    ids=("literal", "f-string", "multiline-f-expression", "template-string"),
)
def test_body_replacement_executes_with_preserved_identity_and_literals(
    tmp_path: Path, source: str, target: str, call: str, body: str, expected: object,
) -> None:
    module_path = tmp_path / "probe.py"
    sibling = "\ndef untouched():\n    return 'sibling'\n"
    original = (
        "import asyncio\nimport json\n" + source + sibling
        + f"\nprint(json.dumps([{call}, untouched()]))\n"
    )
    module_path.write_text(original)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument.from_payload_fields({
        "recipes": [{
            "recipe_id": "body",
            "operations": [{
                "operation": "replace_function_body",
                "file_path": str(module_path),
                "target_qualname": target,
                "body_source": body,
            }],
        }],
    })
    simulation = document.simulate(snapshot)
    assert simulation.is_clean
    assert module_path.read_text() == original
    simulation.apply()
    rewritten = module_path.read_text()
    assert sibling in rewritten
    assert "old_decorator" not in rewritten
    assert "Old body explanation" not in rewritten
    if "# retained header" in original:
        assert "# retained header" in rewritten
    result = subprocess.check_output([sys.executable, str(module_path)], text=True)
    assert json.loads(result) == [expected, "sibling"]


@pytest.mark.parametrize(
    "source",
    (
        "return 1\n",
        "break\n",
        "continue\n",
        "await work()\n",
        "def run():\n    nonlocal missing\n",
        "def run(value, value):\n    pass\n",
    ),
)
def test_simulation_backend_rejects_parsable_but_uncompilable_source(source: str) -> None:
    ast.parse(source)
    with pytest.raises(SyntaxError):
        CodemodBackend.AST_SPAN.validate_source(source, "probe.py")


@pytest.mark.parametrize("body", ("# no statements", "await work()", "nonlocal missing"))
def test_invalid_body_plan_does_not_write_source(tmp_path: Path, body: str) -> None:
    module_path = tmp_path / "probe.py"
    original = "def run():\n    return 1\n"
    module_path.write_text(original)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    document = CodemodPlanDocument.from_payload_fields({
        "recipes": [{
            "recipe_id": "invalid-body",
            "operations": [{
                "operation": "replace_function_body",
                "file_path": str(module_path),
                "target_qualname": "run",
                "body_source": body,
            }],
        }],
    })
    with pytest.raises((ValueError, SyntaxError)):
        document.simulate(snapshot)
    assert module_path.read_text() == original
