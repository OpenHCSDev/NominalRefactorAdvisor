from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.codemod_optional_ast_field import (
    CallKeywordArgDecoder,
    OptionalAstFieldAuthority,
)
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry


def _rewritten(source: str) -> str:
    parsed = SourceModule.from_source_path(Path("m.py"), source).parse()
    function = next(
        node for node in parsed.module.body if isinstance(node, ast.FunctionDef)
    )
    authority = OptionalAstFieldAuthority(
        parsed_module=parsed,
        geometry=SourceTextGeometry(source),
        line_length=88,
        declaration=function,
        decoder=CallKeywordArgDecoder,
    )
    rewritten = source
    for replacement in sorted(
        authority.replacements(), key=lambda item: item.start_offset, reverse=True
    ):
        rewritten = (
            rewritten[: replacement.start_offset]
            + replacement.replacement_source
            + rewritten[replacement.end_offset :]
        )
    ast.parse(rewritten)
    return rewritten


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            "a = any(keyword.arg is None for keyword in call.keywords)",
            "a = bool(CallKeywords.of(call.keywords).splats)",
        ),
        (
            "if x or any(keyword.arg is None for keyword in call.keywords): pass",
            "if x or CallKeywords.of(call.keywords).splats: pass",
        ),
        (
            "a = (x, x or any(keyword.arg is None for keyword in call.keywords))",
            "a = (x, x or bool(CallKeywords.of(call.keywords).splats))",
        ),
        (
            "b = all(keyword.arg is not None for keyword in call.keywords)",
            "b = not CallKeywords.of(call.keywords).splats",
        ),
        (
            "c = [kw.arg for kw in call.keywords if kw.arg is not None and kw.value]",
            "c = [kw.name for kw in CallKeywords.of(call.keywords).named if kw.value]",
        ),
        (
            "d = all(keyword.arg is not None for keyword in call.keywords) == x",
            "d = (not CallKeywords.of(call.keywords).splats) == x",
        ),
    ),
)
def test_expression_probes_read_the_decoder(statement: str, expected: str) -> None:
    rewritten = _rewritten(f"def f(call, x):\n    {statement}\n")
    assert rewritten == f"def f(call, x):\n    {expected}\n"


def test_loop_guards_read_the_decoder() -> None:
    source = (
        "def f(call, projected: tuple[ast.keyword, ...]):\n"
        "    for keyword in call.keywords:\n"
        "        if keyword.arg is None:\n"
        "            continue\n"
        "        use(keyword.arg)\n"
        "    for keyword in projected:\n"
        "        if keyword.arg is None:\n"
        "            return None\n"
        "        use(keyword.arg, keyword.value)\n"
    )
    assert _rewritten(source) == (
        "def f(call, projected: tuple[ast.keyword, ...]):\n"
        "    for keyword in CallKeywords.of(call.keywords).named:\n"
        "        use(keyword.name)\n"
        "    keywords = CallKeywords.of(projected)\n"
        "    if keywords.splats:\n"
        "        return None\n"
        "    for keyword in keywords.named:\n"
        "        use(keyword.name, keyword.value)\n"
    )


def test_probe_outside_a_decoded_collection_is_handed_back() -> None:
    with pytest.raises(ValueError, match="no shape the CallKeywords decoder"):
        _rewritten("def f(keyword):\n    if keyword.arg is None:\n        return 1\n")


def test_element_used_as_an_ast_node_is_handed_back() -> None:
    with pytest.raises(ValueError, match="used as an ast node"):
        _rewritten(
            "def f(call):\n"
            "    return [ast.dump(k) for k in call.keywords if k.arg is not None]\n"
        )


def test_unknown_collection_is_not_a_site() -> None:
    with pytest.raises(ValueError, match="no shape"):
        _rewritten(
            "def f(items):\n    return [k for k in items if k.arg is not None]\n"
        )


def test_filtered_comprehension_is_fit_to_the_line() -> None:
    source = (
        "def f(call_node):\n"
        "    return tuple(\n"
        "        (keyword.arg, keyword.value) for keyword in call_node.keywords"
        " if keyword.arg is not None\n"
        "    )\n"
    )
    rewritten = _rewritten(source)
    assert "CallKeywords.of(call_node.keywords).named" in rewritten
    assert max(len(line) for line in rewritten.splitlines()) <= 88
