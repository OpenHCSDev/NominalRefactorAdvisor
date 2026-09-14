"""Literal operations retain source provenance without fabricating object identity."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceKernel,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(Path("literal.py"), "literal", False, ast.parse(source), source)
    )


@pytest.mark.parametrize(
    "source,release_safe",
    (
        ("item = 1\n", True),
        ("item = None\n", True),
        ("item = 'text'\n", True),
        ("item = b'data'\n", False),
        ("item = [1, 2]\n", False),
        ("item = {'a': (1, 2)}\n", False),
        ("item = {1, 2}\n", False),
    ),
)
def test_literal_capture_proves_member_behavior_without_native_identity(
    source, release_safe
):
    environment = execution(source)
    node = environment.module.module.body[0].value
    result = environment.capture_value(node)
    assert not isinstance(result, (OpenCapturedReference, CapturedNativeObject))
    result.require_closed()
    result.require_class_installation()
    with pytest.raises(ValueError):
        result.require_native_identity(NativeDeclaration(object))
    if release_safe:
        result.require_release()
    else:
        with pytest.raises(ValueError):
            result.require_release()


def test_literal_alias_retains_the_original_evaluated_source_read():
    environment = execution("item = 1\nalias = item\n")
    first, second = environment.module.module.body
    result = environment.capture_value(second.value)
    result.require_class_installation()
    assert result.read.use is environment.source.value_reads_by_node[first.value].use
    assert result.read.context is environment.entry.context


@pytest.mark.parametrize(
    "source",
    (
        "item = [missing]\n",
        "item = custom()\n",
        "unknown()\nitem = 1\n",
    ),
)
def test_literal_extension_does_not_admit_computed_values_or_unknown_prefixes(source):
    environment = execution(source)
    result = environment.capture_value(environment.module.module.body[-1].value)
    with pytest.raises(ValueError):
        result.require_closed()


def test_equal_but_copied_literal_read_is_not_admitted():
    environment = execution("item = 1\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    with pytest.raises(ValueError):
        environment.kernel.read(replace(read, use=replace(read.use)))


def test_kernel_value_query_preserves_the_actual_read_event():
    seen = []

    class Observer(CapturedReferenceKernel):
        def _unproved_value_resolution(self, context):
            seen.append(context)
            return super()._unproved_value_resolution(context)

    environment = execution("item = 1\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    Observer(environment.initial, environment).read(read)
    actual, pending = seen[0]
    assert actual.context is read.context
    assert actual.use is read.use
    assert pending == frozenset()
