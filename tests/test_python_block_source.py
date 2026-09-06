"""Authored suite rendering shares one parse and preserves literal source spans."""

import ast

import pytest

from nominal_refactor_advisor.codemod_declaration_source import ClassMemberSource
from nominal_refactor_advisor.codemod_statement_source import PythonBlockSource


@pytest.mark.parametrize("source", ("", "# comment only\n", "\n\n"))
def test_empty_block_does_not_authorise_a_statement_replacement(source: str) -> None:
    with pytest.raises(ValueError, match="must contain a statement"):
        PythonBlockSource(source).indented_source("    ")


def test_block_cannot_silently_include_code_outside_its_initial_suite() -> None:
    block = PythonBlockSource("    value = 1\nescaped = 2\n")
    with pytest.raises(ValueError, match="escapes its initial indentation"):
        block.indented_source("        ")


def test_member_validation_and_rendering_share_the_parsed_source(monkeypatch) -> None:
    parse = ast.parse
    calls = []

    def counted_parse(source, *args, **kwargs):
        calls.append(source)
        return parse(source, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", counted_parse)
    member = ClassMemberSource.from_source(
        "    def text():\n        return '''first\nlast'''\n",
        indentation="\t",
    )
    assert len(calls) == 1
    namespace = {}
    exec("class Owner:\n" + member.source, namespace)
    assert namespace["Owner"].text() == "first\nlast"


def test_rendering_retains_fstring_text_when_relocating_an_indented_block() -> None:
    block = PythonBlockSource(
        "    def text():\n        return f'''first{2}\n last'''\n"
    )
    namespace = {}
    exec(block.indented_source(""), namespace)
    assert namespace["text"]() == "first2\n last"
