"""Native wrapping retains creation evidence, never raw function identity."""

from dataclasses import replace
from itertools import product
import sys

import pytest

from nominal_refactor_advisor.native_call import NativeDescriptorResult
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import (
    SourceCreatedFunctionCapture,
    SourceNativeDecoratorApplication,
)
from test_source_function_result import execution, function, native


@pytest.mark.parametrize("decorator", ("property", "classmethod", "staticmethod"))
@pytest.mark.parametrize("async_prefix", ("", "async "))
@pytest.mark.parametrize("in_class", (False, True))
def test_native_decorator_result_is_not_the_original_function(
    decorator, async_prefix, in_class
):
    definition = (
        f"@{decorator}\n{async_prefix}def chosen():\n"
        "    raise RuntimeError('body stays deferred')\n"
    )
    source = (
        "class Owner:\n"
        + "".join("    " + line for line in definition.splitlines(True))
        if in_class
        else definition
    )
    slot = "vars(Owner)['chosen']" if in_class else "chosen"
    native(source + f"assert type({slot}) is {decorator}\n")
    environment = execution(source)
    _, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    assert isinstance(result, SourceNativeDecoratorApplication)
    assert isinstance(result, NativeDescriptorResult)
    assert not isinstance(result, SourceCreatedFunctionCapture)
    result.require_closed()
    result.require_class_installation()
    result.require_descriptor_argument()
    assert result.operand is binding.target.decorator_uses[0]
    assert result.argument is result.creation
    for operation in (
        result.source_definition,
        lambda: result.require_definition_identity(binding.target.owner),
        lambda: result.require_native_identity(NativeDeclaration(property)),
        result.require_release,
        result.creation.require_closed,
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize(
    "outer,inner", tuple(product(("property", "classmethod", "staticmethod"), repeat=2))
)
def test_nested_native_decorators_follow_original_application_order(outer, inner):
    source = (
        f"@{outer}\n@{inner}\ndef chosen():\n"
        "    raise RuntimeError('body stays deferred')\n"
    )
    native(source + f"assert type(chosen) is {outer}\n")
    environment = execution(source)
    _, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    result.require_class_installation()
    assert result.operand is binding.target.decorator_uses[0]
    assert result.argument.operand is binding.target.decorator_uses[1]
    assert result.argument.argument is result.creation
    assert result.creation.creation_results == (
        result.creation,
        result.argument,
        result,
    )


@pytest.mark.parametrize("index", (-1, 0, 1, 2))
def test_copied_or_out_of_range_application_is_not_original_evidence(index):
    environment = execution("@property\ndef chosen(): pass\n")
    _, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    copied = replace(result, index=index)
    with pytest.raises(ValueError, match="original creation chain"):
        copied.require_closed()


@pytest.mark.parametrize(
    "decorators",
    (
        "@replacement",
        "@property\n@replacement",
        "@replacement\n@staticmethod",
    ),
)
def test_unknown_application_cannot_inherit_native_result_proof(decorators):
    source = (
        "def replacement(raw):\n    return object\n"
        f"{decorators}\ndef chosen(): pass\n"
    )
    native(source)
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


@pytest.mark.parametrize("future", (False, True))
@pytest.mark.parametrize("decorator", ("property", "classmethod", "staticmethod"))
def test_wrapping_does_not_invent_eager_annotation_execution(future, decorator):
    source = (
        ("from __future__ import annotations\n" if future else "")
        + "events = []\ndef annotation():\n"
        "    events.append('annotation')\n    return object\n"
        f"@{decorator}\ndef chosen(value: annotation()): pass\n"
    )
    deferred = future or sys.version_info >= (3, 14)
    native(source + f"assert events == {[] if deferred else ['annotation']}\n")
    environment = execution(source)
    _, context, binding = function(environment)
    if deferred:
        environment.definition_result(context, binding).require_class_installation()
    else:
        with pytest.raises(ValueError, match="unproved"):
            environment.definition_result(context, binding)
