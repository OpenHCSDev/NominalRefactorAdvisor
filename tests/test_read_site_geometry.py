"""Positioned reads retain exact parser geometry instead of a line-only join."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.value_expression import LexicalValueReference


@pytest.mark.parametrize(
    "source",
    (
        "saved = café.member; café = other; current = café.member\n",
        "def run(café, other):\n"
        "    saved = café.member; café = other; return saved, café.member\n",
        "saved = (café\n    .member); café = other; current = café.member\n",
    ),
    ids=("same-line-module", "same-line-function", "multiline-expression"),
)
def test_each_attribute_read_retains_its_exact_utf8_source_site(source: str) -> None:
    tree = ast.parse(source)
    module = ParsedModule(
        path=Path("read_geometry.py"),
        module_name="read_geometry",
        is_package_init=False,
        module=tree,
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    reads = tuple(
        use
        for use in flow.callable_reference_uses
        if use.target.lexical_reference == LexicalValueReference("café", ("member",))
    )
    nodes = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Attribute)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(reads) == len(nodes) == 2
    assert reads[0].source_span != reads[1].source_span
    assert reads[0].position.dominates(reads[1].position)
    lines = tuple(source.splitlines(keepends=True))
    for use, node in zip(reads, nodes, strict=True):
        assert use.source_span == SourceByteSpan.require_node(node)
        assert use.line == node.lineno
        assert use.source_span.segment(lines) == ast.get_source_segment(source, node)

    saved_alias = next(
        alias for alias in flow.exact_value_aliases if alias.target.root_name == "saved"
    )
    assert saved_alias.source_use is reads[0]


def test_call_target_and_argument_on_one_line_retain_distinct_sites() -> None:
    source = "café.member(café.member)\n"
    module = ParsedModule(
        path=Path("read_geometry.py"),
        module_name="read_geometry",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    flow = compact_product_flow_projection(module).flows[-1]
    (call,) = flow.calls
    (argument_read,) = (
        use
        for use in flow.callable_reference_uses
        if use.target.lexical_reference == call.target.lexical_reference
    )
    assert call.target_use.source_span != argument_read.source_span
    assert call.target_use.position.dominates(argument_read.position)
    assert call.line == call.target_use.line == argument_read.line == 1
