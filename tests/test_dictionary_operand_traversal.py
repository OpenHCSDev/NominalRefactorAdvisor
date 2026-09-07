"""Dictionary operand order is shared by flow and lexical dependency analysis."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
    _DeclarationDependencyCollector,
)
from nominal_refactor_advisor.lexical_bindings import DictionaryEvaluationVisitor
from nominal_refactor_advisor.product_flow import (
    _CompactFlowCollector,
    LexicalValueReference,
    compact_product_flow_projection,
)


def _module(source: str) -> ParsedModule:
    return ParsedModule(
        Path("pkg/dict_order.py"), "pkg.dict_order", False, ast.parse(source), source
    )


@pytest.mark.parametrize(
    "expression, expected",
    (
        (
            "{first_key: first_value, second_key: second_value}",
            ("first_key", "first_value", "second_key", "second_value"),
        ),
        (
            "{first_key: first_value, **middle, last_key: last_value}",
            ("first_key", "first_value", "middle", "last_key", "last_value"),
        ),
        (
            "{**first, next_key: next_value, **last}",
            ("first", "next_key", "next_value", "last"),
        ),
        (
            "{outer_key: {inner_key: inner_value}, last_key: last_value}",
            ("outer_key", "inner_key", "inner_value", "last_key", "last_value"),
        ),
    ),
)
def test_compact_reads_follow_key_value_pairs(expression, expected):
    projection = compact_product_flow_projection(_module("result = " + expression))
    reads = sorted(
        projection.reference_reads_by_span.values(),
        key=lambda read: read.use.position.event_index,
    )
    assert tuple(read.use.target.lexical_reference for read in reads) == tuple(
        LexicalValueReference(name) for name in expected
    )


def test_flow_and_dependency_visitors_share_the_dictionary_method():
    assert _CompactFlowCollector.visit_Dict is DictionaryEvaluationVisitor.visit_Dict
    assert (
        _DeclarationDependencyCollector.visit_Dict
        is DictionaryEvaluationVisitor.visit_Dict
    )


def test_dependency_projection_preserves_walrus_binding_order():
    source = (
        "key = property\n"
        "class Owner:\n"
        "    result = {key: (key := object), key: None}\n"
    )
    parsed = _module(source)
    before, after = parsed.module.body[1].body[0].value.keys
    projection = ModuleLexicalDependencyProjection.from_module(parsed.module)
    references = tuple(surface.reference for surface in projection.direct_name_surfaces)
    assert any(reference is before for reference in references)
    assert all(reference is not after for reference in references)
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + "\nassert tuple(Owner.result) == (property, object)",
        ],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr


def test_earlier_value_call_precedes_later_key_read():
    source = (
        "key = property\n"
        "def change():\n"
        "    global key\n"
        "    key = object\n"
        "    return None\n"
        "result = {'literal': change(), key: None}\n"
    )
    projection = compact_product_flow_projection(_module(source))
    key_read = next(
        read
        for read in projection.reference_reads_by_span.values()
        if read.use.target.lexical_reference == LexicalValueReference("key")
    )
    assert key_read.context.flow.calls[0].position.dominates(key_read.use.position)
    native = subprocess.run(
        [sys.executable, "-c", source + "\nassert tuple(result)[1] is object"],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr


def test_native_key_value_and_unpack_order():
    source = """import json
events = []
def key(name):
    events.append('key:' + name)
    return name
def value(name):
    events.append('value:' + name)
    return name
class Mapping:
    def keys(self):
        events.append('unpack:keys')
        return ['middle']
    def __getitem__(self, name):
        events.append('unpack:item')
        return name
result = {key('first'): value('first'), **Mapping(), key('last'): value('last')}
print(json.dumps(events))
"""
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert json.loads(native.stdout) == [
        "key:first",
        "value:first",
        "unpack:keys",
        "unpack:item",
        "key:last",
        "value:last",
    ]
