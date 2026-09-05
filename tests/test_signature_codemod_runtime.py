"""Exercise authored signature edits through source replay and fresh Python runs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    FunctionSignatureSourceAuthority,
)
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry


def test_historical_renderer_helpers_can_be_extracted_as_one_dsl_batch(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    source = (root / "tests/fixtures/renderer_helper_before.py").read_text()
    module_path = tmp_path / "nominal_refactor_advisor/codemod.py"
    module_path.parent.mkdir()
    module_path.write_text(source)
    expected = subprocess.check_output([sys.executable, str(module_path)], text=True)
    plan_path = root / "docs/examples/renderer_extraction_sequence.json"
    cli_result = subprocess.run(
        [
            sys.executable, "-m", "nominal_refactor_advisor", str(tmp_path),
            "--codemod-plan", str(plan_path), "--codemod-simulate", "--json",
        ],
        cwd=root, capture_output=True, text=True, check=True,
    )
    cli_report = json.loads(cli_result.stdout)
    assert cli_report["plan_sequence_simulation"]["is_clean"]
    assert cli_report["applied"] is False
    assert module_path.read_text() == source
    sequence = CodemodPlanSequence.from_payload_fields(json.loads(plan_path.read_text()))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = sequence.simulate(snapshot)
    assert simulation.is_clean
    assert simulation.stage_count == 4
    assert module_path.read_text() == source
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(module_path)], text=True) == expected
    module = ast.parse(module_path.read_text())
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    ancestor = classes["RendererSourceAuthority"]
    child = classes["DirectBuildFindingRendererFindingRecipeSynthesizer"]
    assert [node.name for node in ancestor.body if isinstance(node, ast.FunctionDef)] == [
        "renderer_lambda", "renderer_source",
    ]
    assert not any(isinstance(node, ast.FunctionDef) for node in child.body)
    assert [ast.unparse(base) for base in child.bases] == ["RendererSourceAuthority"]


@pytest.mark.parametrize(
    "original_suffix",
    (
        "(value=3):",
        "(\n        value=3,\n    ) -> int:",
        "(\n        value=(lambda: 3)(),\n    ) -> (lambda: int):",
        '(value=3) -> dict[str, "colon: and )"]:',
        "(value=3) -> (\n        dict[str, int]\n    ):",
    ),
    ids=("inline", "multiline", "lambda-colons", "annotation-string", "annotation-lines"),
)
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_saved_signature_plan_preserves_runtime_and_untouched_source(
    tmp_path: Path, original_suffix: str, asynchronous: bool,
) -> None:
    module_path = tmp_path / "probe.py"
    prefix = "async " if asynchronous else ""
    call = "asyncio.run(Worker.café())" if asynchronous else "Worker.café()"
    before_header = f"{prefix}def café{original_suffix}"
    after_header = f"{prefix}def café(value: int = 3) -> int:"
    source = (
        "from __future__ import annotations\n"
        "import asyncio\n"
        "import json\n"
        "class Worker:\n"
        "    @staticmethod\n"
        f"    {before_header} return value  # keep café's inline suite\n"
        f"print(json.dumps({call}))\n"
    )
    module_path.write_text(source, encoding="utf-8")
    expected = subprocess.check_output([sys.executable, str(module_path)], text=True)
    document = CodemodPlanDocument.from_payload_fields(json.loads(json.dumps({
        "recipes": [{
            "recipe_id": "signature",
            "operations": [{
                "operation": "replace_function_signature",
                "file_path": str(module_path),
                "target_qualname": "Worker.café",
                "signature_suffix": "(value: int = 3) -> int:",
            }],
        }],
    })))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = document.simulate(snapshot)
    assert simulation.is_clean
    assert module_path.read_text(encoding="utf-8") == source
    simulation.apply()
    assert module_path.read_text(encoding="utf-8") == source.replace(
        before_header, after_header,
    )
    assert subprocess.check_output([sys.executable, str(module_path)], text=True) == expected


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_signature_replacement_preserves_block_body_and_newlines(newline: str) -> None:
    original = "(\n    value=3,\n) -> int:"
    replacement = "(value: int = 3) -> int:"
    source = (
        f"def café{original}  # retain this comment\n"
        '    """Retain this docstring."""\n'
        "    # Retain this body comment.\n"
        "    return value\n"
    ).replace("\n", newline)
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    authority = FunctionSignatureSourceAuthority(node=node, source=source)
    rewritten = authority.geometry.source_with_replacements_in_span(
        0, len(source), (authority.replacement(replacement),),
    )
    assert rewritten == source.replace(original.replace("\n", newline), replacement)


@pytest.mark.parametrize(
    "suffix",
    (
        "(\n    value=3,  # parameter explanation\n):",
        "(value=3) -> (\n    int  # return explanation\n):",
    ),
)
def test_signature_replacement_refuses_to_discard_header_comments(suffix: str) -> None:
    source = f"def run{suffix}\n    return value\n"
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    authority = FunctionSignatureSourceAuthority(node=node, source=source)
    with pytest.raises(ValueError, match="would discard comments"):
        authority.replacement("(value=3):")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 needs Python 3.12")
def test_signature_geometry_keeps_generic_type_parameters() -> None:
    source = "def run[T: tuple[int, (str)]](value: T) -> T: return value\n"
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    geometry = SourceTextGeometry(source)
    assert geometry.function_parameter_span(node).source_text(source) == "value: T"
    assert geometry.function_signature_suffix_span(node).source_text(source) == (
        "(value: T) -> T:"
    )
    authority = FunctionSignatureSourceAuthority(node=node, source=source)
    rewritten = geometry.source_with_replacements_in_span(
        0, len(source), (authority.replacement("(value: T, *, flag=False) -> T:"),),
    )
    assert rewritten == source.replace("(value: T)", "(value: T, *, flag=False)")
