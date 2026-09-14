"""Dictionary operand order is shared and does not confer assembly proof."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
    _DeclarationDependencyCollector,
)
from nominal_refactor_advisor.lexical_bindings import DictionaryEvaluationVisitor
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import (
    CompactFunctionCall,
    CompactMutation,
    _CompactFlowCollector,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def module(source):
    return ParsedModule(
        Path("/repo/dict_order.py"), "dict_order", False, ast.parse(source), source
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
def test_compact_original_reads_follow_key_value_pairs(expression, expected):
    observed = source_product_flow_projection(module("result = " + expression + "\n"))
    reads = sorted(
        observed.reference_reads_by_node.items(),
        key=lambda pair: pair[1].use.position.event_index,
    )
    assert tuple(node.id for node, read in reads) == expected
    assert all(
        read.context is observed.context_for_owner(read.context.flow.owner)
        for node, read in reads
    )


def test_flow_and_dependency_visitors_share_the_dictionary_method():
    assert _CompactFlowCollector.visit_Dict is DictionaryEvaluationVisitor.visit_Dict
    assert (
        _DeclarationDependencyCollector.visit_Dict
        is DictionaryEvaluationVisitor.visit_Dict
    )


def test_dependency_projection_keeps_pre_walrus_key_external_and_later_key_local():
    source = (
        "key = property\n"
        "class Owner:\n"
        "    result = {key: (key := object), key: None}\n"
    )
    parsed = module(source)
    dictionary = parsed.module.body[1].body[0].value
    before, after = dictionary.keys
    projection = ModuleLexicalDependencyProjection.from_module(parsed.module)
    references = tuple(surface.reference for surface in projection.direct_name_surfaces)
    assert any(reference is before for reference in references)
    assert all(reference is not after for reference in references)
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + "\nassert tuple(Owner.result) == (property, object)\n",
        ],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        "change()",
        "(key := object)",
    ),
)
def test_earlier_value_cannot_leave_a_stale_later_key_capture(mutation):
    source = (
        "key = property\n"
        "def change():\n"
        "    global key\n"
        "    key = object\n"
        "    return None\n"
        f"result = {{'literal': {mutation}, key: None}}\n"
    )
    parsed = module(source)
    env = SourceModuleExecution.from_module(parsed)
    dictionary = parsed.module.body[-1].value
    later_key = dictionary.keys[1]
    capture = env.capture(later_key)
    # Dictionary assembly is not proved for referenced keys. The actual earlier
    # mutation must not disappear simply because AST fields group keys first.
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(ValueError):
        capture.require_native((NativeDeclaration(property),))
    native = subprocess.run(
        [sys.executable, "-c", source + "\nassert tuple(result)[1] is object\n"],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr
    key_read = env.source.reference_reads_by_node[later_key]
    if isinstance(dictionary.values[0], ast.NamedExpr):
        operation = next(
            op
            for op in env.source.operations_by_node[dictionary.values[0].target]
            if isinstance(op.event, CompactMutation)
        )
    else:
        operation = next(
            op
            for op in env.source.operations_by_node[dictionary.values[0]]
            if isinstance(op.event, CompactFunctionCall)
        )
    assert operation.position.dominates(key_read.use.position)


def test_unpack_must_complete_before_the_following_key_can_be_used():
    source = "key = property\nresult = {**object, key: None}\n"
    parsed = module(source)
    env = SourceModuleExecution.from_module(parsed)
    dictionary = parsed.module.body[-1].value
    capture = env.capture(dictionary.keys[1])
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(ValueError):
        capture.require_closed()
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert native.returncode != 0
    assert "not a mapping" in native.stderr


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
