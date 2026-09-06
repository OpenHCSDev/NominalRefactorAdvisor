"""Whole-declaration edits preserve lexical ownership and authored literal bytes."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceTargetOperation,
    SourceRewriteTarget,
)


def replace_method(source: str, payload: str, qualname: str = "Owner.run") -> str:
    module = ParsedModule(
        path=Path("layout.py"),
        module_name="layout",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceTargetOperation(
                target=SourceRewriteTarget(file_path="layout.py", qualname=qualname),
                replacement_source=payload,
            ),
        )
    )
    result = plan.simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert result.is_clean
    return result.final_snapshot.sources_by_file_path["layout.py"]


@pytest.mark.parametrize("target_indent", ("    ", "\t"))
@pytest.mark.parametrize("payload_indent", ("", "    ", "\t"))
@pytest.mark.parametrize("decorated", (False, True))
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_replacement_derives_owner_indentation_without_changing_literals(
    target_indent: str,
    payload_indent: str,
    decorated: bool,
    newline: str,
) -> None:
    source = (
        "class Owner:\n"
        f"{target_indent}marker = 1\n"
        + (f"{target_indent}@staticmethod\n" if decorated else "")
        + f"{target_indent}def run():\n{target_indent}    return 'old'\n"
    )
    payload = (
        "# retained leading comment\n"
        f"{payload_indent}def run():\n"
        f"{payload_indent}    return '''first\n"
        "  protected literal indentation\n"
        "end''' # retained trailing comment\n"
    )
    rewritten = replace_method(
        source.replace("\n", newline), payload.replace("\n", newline)
    )
    namespace = {}
    exec(rewritten, namespace)
    owner = namespace["Owner"]
    assert "run" in vars(owner)
    assert "run" not in namespace
    assert owner.run() == "first\n  protected literal indentation\nend"
    assert owner.marker == 1
    assert rewritten.count("# retained leading comment") == 1
    assert rewritten.count("# retained trailing comment") == 1
    assert rewritten.count("@staticmethod") == int(decorated)


def test_nested_class_replacement_remains_in_its_declared_owner() -> None:
    source = "class Owner:\n    marker = 1\n    class Inner:\n        old = True\n"
    rewritten = replace_method(source, "class Inner:\n    new = True\n", "Owner.Inner")
    namespace = {}
    exec(rewritten, namespace)
    assert namespace["Owner"].Inner.new is True
    assert "Inner" not in namespace


def test_nested_async_replacement_remains_in_its_function() -> None:
    source = (
        "def owner():\n    marker = 1\n    async def run(): return 1\n    return run\n"
    )
    rewritten = replace_method(source, "async def run():\n    return 2\n", "owner.run")
    namespace = {}
    exec(rewritten, namespace)
    assert namespace["owner"]().__qualname__ == "owner.<locals>.run"
    assert "run" not in namespace
