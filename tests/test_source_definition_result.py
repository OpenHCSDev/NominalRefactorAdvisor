"""Exact creation results are distinct from declaration and later-slot selection."""

import ast
import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import (
    SourceCreatedClassCapture,
    SourceModuleExecution,
)


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("definition_result.py"),
            "definition_result",
            False,
            ast.parse(source),
            source,
        )
    )


def definition(environment, index=0):
    node = environment.module.module.body[index]
    operation = environment.definition_operation(node)
    return environment.context_for_owner(operation.owner), operation.event


def test_actual_plain_definition_result_has_exact_owner():
    environment = execution("class Original: pass\n")
    context, binding = definition(environment)
    result = environment.definition_result(context, binding)
    assert isinstance(result, SourceCreatedClassCapture)
    assert result.entry.definition is binding
    result.require_definition_identity(binding.target.owner)


@pytest.mark.parametrize(
    "kind,rejection",
    (
        ("decorator", "Class decorator result remains unproved"),
        (
            "metaclass",
            "Native object identity remains open: unproved_execution_effects",
        ),
    ),
)
def test_native_replacing_construction_does_not_claim_raw_definition(kind, rejection):
    prefix = "class Original: pass\n"
    if kind == "decorator":
        source = (
            prefix + "def replace(raw): return Original\n@replace\nclass Other: pass\n"
        )
    else:
        source = prefix + (
            "class Replace(type):\n"
            "    def __new__(meta, name, bases, namespace): return Original\n"
            "class Other(metaclass=Replace): pass\n"
        )
    native = subprocess.run(
        [sys.executable, "-c", source + "assert Other is Original\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr
    environment = execution(source)
    context, binding = definition(environment, -1)
    with pytest.raises(ValueError, match=rejection):
        environment.definition_result(context, binding)


def test_closed_native_identity_does_not_prove_source_definition():
    value = CapturedNativeObject(type)
    value.require_closed()
    environment = execution("class Original: pass\n")
    _, binding = definition(environment)
    with pytest.raises(ValueError, match="definition identity remains unproved"):
        value.require_definition_identity(binding.target.owner)


def test_definition_identity_preserves_the_original_open_capture_cause():
    environment = execution("class Original: pass\n")
    _, binding = definition(environment)
    cause = ValueError("original binding evidence unavailable")
    value = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_BINDING, cause=cause
    )
    with pytest.raises(CapturedReferenceRejection) as caught:
        value.require_definition_identity(binding.target.owner)
    assert caught.value.violation is CapturedReferenceViolation.UNPROVED_BINDING
    assert caught.value.__cause__ is cause


def test_source_result_rejects_equal_but_foreign_definition_owner():
    environment = execution("class Original: pass\n")
    context, binding = definition(environment)
    result = environment.definition_result(context, binding)
    other_owner = replace(binding.target.owner)
    assert other_owner == binding.target.owner
    assert other_owner is not binding.target.owner
    with pytest.raises(ValueError, match="another source definition"):
        result.require_definition_identity(other_owner)


@pytest.mark.parametrize("forged", ("context", "binding", "reparsed"))
def test_query_rejects_noncanonical_event_or_context(forged):
    environment = execution("class Original: pass\n")
    context, binding = definition(environment)
    if forged == "context":
        context = replace(context)
    elif forged == "binding":
        binding = replace(binding)
    else:
        context, binding = definition(execution(environment.module.source))
    with pytest.raises(ValueError, match="canonical context|unique original operation"):
        environment.definition_result(context, binding)


def test_equal_span_copied_node_cannot_select_definition_event():
    environment = execution("class Original: pass\n")
    original = environment.module.module.body[0]
    copied = copy.deepcopy(original)
    assert ast.dump(copied, include_attributes=True) == ast.dump(
        original, include_attributes=True
    )
    with pytest.raises(ValueError, match="unique actual operation"):
        environment.definition_operation(copied)


def test_actual_nondefinition_event_does_not_enter_definition_dispatch():
    environment = execution("value = 1\n")
    context = environment.entry.context
    binding = context.flow.mutations[-1]
    with pytest.raises(ValueError, match="actual definition binding"):
        environment.definition_result(context, binding)


def test_plain_function_result_does_not_claim_body_activation():
    environment = execution("def original(): unknown()\n")
    context, binding = definition(environment)
    result = environment.definition_result(context, binding)
    result.require_definition_identity(binding.target.owner)
    assert result.source_definition() == (context, binding)
    with pytest.raises(ValueError, match="unproved_execution_effects"):
        environment.required_prefix(
            result.context, result.context.flow.calls[0].position
        )


def test_alias_value_capture_keeps_earlier_definition_before_rebinding():
    source = "class Original: pass\nsaved = Original\nOriginal = None\n"
    environment = execution(source)
    context, binding = definition(environment)
    alias = environment.module.module.body[1]
    result = environment.capture_value(alias.value)
    result.require_definition_identity(binding.target.owner)
    assert result.entry.definition is binding
    # Query the original creation, not the later current value of Original.
    environment.definition_result(context, binding).require_definition_identity(
        binding.target.owner
    )
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + "assert isinstance(saved, type)\nassert Original is None\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


@pytest.mark.parametrize(
    "previous, admitted",
    (("Original = 1\n", True), ("class Original: pass\n", False)),
)
def test_creation_result_requires_final_binding_release_admission(previous, admitted):
    environment = execution(previous + "class Original: pass\n")
    context, binding = definition(environment, -1)
    if admitted:
        result = environment.definition_result(context, binding)
        result.require_definition_identity(binding.target.owner)
    else:
        with pytest.raises(ValueError):
            environment.definition_result(context, binding)


def test_dataclass_prefix_remains_explicitly_unproved():
    environment = execution(
        "from dataclasses import dataclass\n@dataclass\n"
        "class Product:\n    left: object\n    right: object\n"
        "class Original: pass\n"
    )
    context, binding = definition(environment, -1)
    with pytest.raises(ValueError, match="unproved_execution_effects"):
        environment.definition_result(context, binding)
