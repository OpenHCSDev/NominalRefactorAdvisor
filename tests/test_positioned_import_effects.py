"""Import aliases are separate actual operations, not one replayed statement."""

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactImportTarget, CompactMutation
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(Path("imports.py"), "imports", False, ast.parse(source), source)
    )


def imports(environment):
    return tuple(
        site
        for site in environment.source.operations
        if isinstance(site.event, CompactMutation)
        and isinstance(site.event.target, CompactImportTarget)
    )


@pytest.mark.parametrize(
    "statement,expression,expected",
    (
        ("from builtins import object, property", "property", property),
        ("from builtins import property as first, object as second", "first", property),
        ("from builtins import property as same, object as same", "same", object),
        ("import builtins, typing", "builtins.property", property),
        ("from typing import Annotated, Literal, get_type_hints", "property", property),
    ),
)
def test_multi_imports_close_incrementally(statement, expression, expected):
    source = statement + "\nselected = " + expression + "\n"
    environment = execution(source)
    for operation in imports(environment):
        context = environment.context_for_owner(operation.owner)
        environment.required_prefix(context, operation.position)
        environment.require_import_operation(operation)
    capture = environment.capture_value(environment.module.module.body[-1].value)
    assert (
        capture.require_native_identity(NativeDeclaration(expected)).declaration
        is expected
    )


def test_later_failed_import_does_not_poison_the_earlier_binding():
    source = "from builtins import property as first, definitely_missing as second\n"
    # Native import leaves the first binding installed when the later import fails.
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            "namespace = {}\ntry:\n    exec("
            + repr(source)
            + ", namespace)\nexcept ImportError:\n    assert namespace['first'] is property\n"
            + "else:\n    raise AssertionError('expected missing import')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert native.returncode == 0, native.stderr
    environment = execution(source)
    first, second = imports(environment)
    environment.require_import_operation(first)
    context = environment.context_for_owner(second.owner)
    environment.required_prefix(context, second.position)
    with pytest.raises(ValueError):
        environment.require_import_operation(second)
    # A whole-statement check must still reject the missing later binding.
    with pytest.raises(ValueError):
        environment.require_import(environment.module.module.body[0])


def test_equal_operation_copy_is_not_the_original_import_observation():
    environment = execution("from builtins import property, object\n")
    first, _ = imports(environment)
    with pytest.raises(ValueError, match="original source operation"):
        environment.require_import_operation(replace(first))


def test_foreign_same_source_operation_cannot_supply_import_evidence():
    source = "from builtins import property, object\n"
    environment, foreign = execution(source), execution(source)
    with pytest.raises(ValueError, match="actual flow context"):
        environment.require_import_operation(imports(foreign)[0])


def test_multi_import_does_not_skip_an_earlier_unknown_effect():
    environment = execution(
        "unknown()\nfrom builtins import property, object\nselected = property\n"
    )
    capture = environment.capture_value(environment.module.module.body[-1].value)
    assert isinstance(capture, OpenCapturedReference)
    with pytest.raises(ValueError):
        capture.require_closed()
