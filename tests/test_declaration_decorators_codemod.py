"""One decorator authority supports class and function source without suite rewrites."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC,
    DeclarationDecoratorsSourceAuthority,
    FunctionMutationOperationABC,
    FunctionDecoratorsSourceAuthority,
    ReplaceDeclarationDecoratorsOperation,
    ReplaceFunctionDecoratorsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("indent", ("    ", "\t"))
def test_cli_changes_nested_class_decorators_without_rewriting_body(
    tmp_path: Path,
    newline: str,
    indent: str,
) -> None:
    path = tmp_path / "probe.py"
    suffix = (
        f"{indent}class Item: # preserve header\n"
        f"{indent}{indent}value: int\n"
        f"{indent}{indent}text = '''café\n  literal indentation\n'''\n"
        "print(Outer.Item(3) == Outer.Item(3))\n"
        "print(ascii(Outer.Item.text))"
    ).replace("\n", newline)
    source = (
        "from dataclasses import dataclass\nclass Outer:\n"
        f"{indent}# retain explanation\n"
        f"{indent}@(\n{indent}{indent}dataclass\n{indent})\n"
    ).replace("\n", newline) + suffix
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Outer.Item"
                ),
                decorators_source="@dataclass(eq=False)",
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
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = subprocess.check_output([sys.executable, str(path)], text=True)
    assert before.splitlines()[0] == "True"
    assert after.splitlines()[0] == "False"
    assert before.splitlines()[1:] == after.splitlines()[1:]
    assert path.read_bytes().decode("utf-8").endswith(suffix)
    assert "# retain explanation" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("prefix", ("", "@first\n@second\n"))
@pytest.mark.parametrize("replacement", ("", "@third"))
@pytest.mark.parametrize(
    "declaration", ("class Owner: pass", "def Owner(): pass", "async def Owner(): pass")
)
def test_shared_operation_adds_replaces_and_removes_decorators(
    tmp_path: Path,
    prefix: str,
    replacement: str,
    declaration: str,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(prefix + declaration, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
                decorators_source=replacement,
            ),
        )
    )
    plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    ).apply()
    assert path.read_bytes().decode("utf-8") == (
        (replacement + "\n" if replacement else "") + declaration
    )


@pytest.mark.parametrize(
    "source, qualname, payload",
    (
        ("value = 1\n", None, "@dataclass"),
        ("class Owner: pass\n", "Owner", "value = 2"),
        ("class Owner: pass\n", "Owner", "class Extra: pass"),
        (
            "@decorate(\n    # retain\n    option\n)\nclass Owner: pass\n",
            "Owner",
            "@dataclass",
        ),
    ),
)
def test_declaration_decorators_reject_unowned_surfaces(
    tmp_path: Path,
    source: str,
    qualname: str | None,
    payload: str,
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname=qualname
                ),
                decorators_source=payload,
            ),
        )
    )
    with pytest.raises(ValueError):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode("utf-8")


def test_function_operation_remains_a_strict_nominal_refinement(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text("class Owner: pass\n", encoding="utf-8", newline="")
    operation = ReplaceFunctionDecoratorsOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Owner"),
        decorators_source="@dataclass",
    )
    assert isinstance(operation, DeclarationDecoratorsPayload)
    assert isinstance(operation, DeclarationMutationOperationABC)
    assert isinstance(operation, FunctionMutationOperationABC)
    assert (
        FunctionDecoratorsSourceAuthority.replacement
        is DeclarationDecoratorsSourceAuthority.replacement
    )
    with pytest.raises(ValueError, match="function declaration"):
        CodemodPlanSequence.from_operations((operation,)).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
