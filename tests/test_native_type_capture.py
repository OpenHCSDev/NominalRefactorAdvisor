"""Exact type evidence does not imply runtime identity or callable behavior."""

import ast
from inspect import get_annotations
from pathlib import Path
from types import FunctionType
from typing import ClassVar

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CreatedNamespaceDictionary,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(Path("types.py"), "types", False, ast.parse(source), source)
    )


@pytest.mark.parametrize(
    "definition,native_type",
    (
        ("def value():\n    raise RuntimeError('must not execute')\n", FunctionType),
        ("async def value():\n    yield missing\n", FunctionType),
        ("class value:\n    pass\n", type),
        ("value = {'key': 1}\n", dict),
        ("value = globals()\n", dict),
    ),
)
def test_source_values_share_exact_type_installation_proof(definition, native_type):
    environment = execution(definition + "alias = value\n")
    value = environment.capture_value(environment.module.module.body[-1].value)
    value.require_class_installation()
    assert value.native_type is native_type
    assert (
        type(value).require_class_installation
        is CapturedNativeObject.require_class_installation
    )
    # Knowing a raw function's type does not authorize executing its body,
    # nor does it create an analyzer-side object standing in for its result.
    assert not isinstance(value, CapturedNativeObject)


@pytest.mark.parametrize(
    "source",
    (
        "def value():\n    return unknown()\nclass Owner:\n    member = value\n",
        "class value:\n    pass\nclass Owner:\n    member = value\n",
    ),
)
def test_raw_definitions_can_be_installed_without_invocation(source):
    environment = execution(source)
    environment.require_class_creation(environment.module.module.body[-1])


@pytest.mark.parametrize(
    "source",
    (
        "@replacement\ndef value(): pass\nalias = value\n",
        "class value(metaclass=custom): pass\nalias = value\n",
    ),
)
def test_unknown_definition_results_do_not_acquire_raw_native_type(source):
    environment = execution(source)
    with pytest.raises(ValueError):
        value = environment.capture_value(environment.module.module.body[-1].value)
        value.require_class_installation()


def test_shared_type_proof_honors_capture_closure():
    class Unclosed(CapturedNativeObject):
        def require_closed(self):
            raise ValueError("capture remains open")

    with pytest.raises(ValueError, match="capture remains open"):
        Unclosed(1).require_class_installation()


def test_native_type_declaration_annotation_resolves():
    assert (
        get_annotations(CreatedNamespaceDictionary, eval_str=True)["native_type"]
        == ClassVar[type]
    )
