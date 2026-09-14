"""Plain function creation is distinct from invocation and later binding reads."""

import ast
import copy
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.source_execution import (
    SourceCreatedClassCapture,
    SourceCreatedFunctionCapture,
    SourceDefinitionEntry,
    SourceModuleExecution,
)


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("function_result.py"),
            "function_result",
            False,
            ast.parse(source),
            source,
        )
    )


def function(environment, name="chosen"):
    node = next(
        item
        for item in ast.walk(environment.module.module)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    operation = environment.definition_operation(node)
    return node, environment.context_for_owner(operation.owner), operation.event


def native(source):
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "header,body",
    (
        ("def chosen():", "raise RuntimeError('body must remain deferred')"),
        ("async def chosen():", "raise RuntimeError('body must remain deferred')"),
        ("def chosen():", "yield missing()"),
        ("async def chosen():", "yield missing()"),
        ("def chosen(value=1, *, option=None):", "return missing(value)"),
        ("def chosen(value: object) -> object:", "return missing(value)"),
    ),
)
def test_plain_result_proves_creation_without_invoking_body(header, body):
    source = header + "\n    " + body + "\nalias = chosen\n"
    native(
        source
        + "import types\nassert type(chosen) is types.FunctionType\nassert alias is chosen\n"
    )
    environment = execution(source)
    node, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    assert isinstance(result, SourceCreatedFunctionCapture)
    assert result.node is node
    assert result.execution is environment
    assert tuple(item.name for item in fields(result)) == ("execution", "node")
    result.require_closed()
    result.require_definition_identity(binding.target.owner)
    assert result.source_definition() == (context, binding)
    assert result.source_definition()[1] is binding
    captured_alias = environment.capture_value(environment.module.module.body[-1].value)
    captured_alias.require_definition_identity(binding.target.owner)
    # Creation does not supply a callable execution, release, or class-base proof.
    with pytest.raises(ValueError):
        result.require_release()
    with pytest.raises(ValueError):
        environment.required_prefix(
            result.context, result.context.flow.calls[0].position
        )
    # Never execute a source body to acquire its result or runtime identity.


@pytest.mark.parametrize("future", (False, True))
def test_annotation_callback_obeys_native_phase_without_losing_inventory(future):
    source = (
        ("from __future__ import annotations\n" if future else "")
        + "events = []\ndef annotation():\n    events.append('annotation')\n    return object\n"
        + "def chosen(value: annotation()):\n    return value\n"
    )
    native(
        source
        + f"assert events == {[] if future or sys.version_info >= (3, 14) else ['annotation']}\n"
        + "annotations = chosen.__annotations__\n"
        + f"assert events == {[] if future else ['annotation']}\n"
    )
    environment = execution(source)
    _, context, binding = function(environment)
    if future or sys.version_info >= (3, 14):
        # Inventory retains deferred obligations, but definition execution must
        # not invent a running event for a native deferred annotation callback.
        annotation = environment.module.module.body[-1].args.args[0].annotation
        assert any(
            site.trigger is annotation
            for site in environment.effects.sites_by_owner[environment.module.module]
        )
        assert environment.source.operations_by_node.get(annotation, ()) == ()
        result = environment.definition_result(context, binding)
        result.require_definition_identity(binding.target.owner)
    else:
        with pytest.raises(ValueError, match="unproved_execution_effects"):
            environment.definition_result(context, binding)


@pytest.mark.parametrize("future", (False, True))
def test_default_callback_result_does_not_bypass_header_effect_proof(future):
    source = (
        ("from __future__ import annotations\n" if future else "")
        + "def replace():\n    global chosen\n    chosen = object\n    return None\n"
        "def chosen(value=replace()):\n    return value\n"
    )
    native(source + "import types\nassert type(chosen) is types.FunctionType\n")
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


@pytest.mark.parametrize("future", (False, True))
def test_decorator_can_replace_the_actual_raw_function(future):
    source = (
        "from __future__ import annotations\n" if future else ""
    ) + "def replace(raw):\n    return object\n@replace\ndef chosen():\n    pass\n"
    native(source + "assert chosen is object\n")
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="decorator result remains unproved"):
        environment.definition_result(context, binding)


@pytest.mark.parametrize(
    "source",
    (
        "if False:\n    def chosen(): pass\n",
        "if True:\n    def chosen(): pass\n",
        "for item in (1, 2):\n    def chosen(): pass\n",
        "def outer():\n    def chosen(): pass\n    return chosen\n",
    ),
)
def test_unadmitted_or_repeated_creator_does_not_acquire_module_identity(source):
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


def test_class_body_creator_retains_its_actual_nonmodule_identity():
    source = "class Owner:\n    def chosen():\n        raise RuntimeError('deferred')\n"
    native(source)
    environment = execution(source)
    node, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    assert isinstance(result, SourceCreatedFunctionCapture)
    assert result.node is node
    result.require_closed()
    actual_context, actual_binding = result.source_definition()
    assert actual_context is context
    assert actual_context is not environment.entry.context
    assert actual_binding is binding


def test_class_creator_rejects_equal_but_nonoriginal_code_receipt():
    environment = execution("class Owner:\n    def chosen(): pass\n")
    owner = environment.class_entry(environment.module.module.body[0])
    original = owner.capture.body
    owner.require_native_creator(original)
    copied = replace(original)
    assert copied == original and copied is not original
    with pytest.raises(ValueError, match="different class body"):
        owner.require_native_creator(copied)


@pytest.mark.parametrize("later", ("chosen = object\n", "del chosen\n"))
def test_historical_alias_capture_remains_separate_from_later_effect_closure(later):
    source = "def chosen(): pass\nsaved = chosen\n" + later + "result = saved\n"
    native(source + "import types\nassert type(result) is types.FunctionType\n")
    environment = execution(source)
    _, context, binding = function(environment)
    historical = environment.capture_value(environment.module.module.body[1].value)
    historical.require_definition_identity(binding.target.owner)
    environment.definition_result(context, binding).require_definition_identity(
        binding.target.owner
    )
    result = environment.capture_value(environment.module.module.body[-1].value)
    # The live saved slot proves retention at either mutation's actual cut.
    result.require_definition_identity(binding.target.owner)


def test_plain_function_does_not_silently_admit_a_prior_dataclass_protocol():
    environment = execution(
        "from dataclasses import dataclass\n@dataclass\nclass Product:\n"
        "    left: object\n    right: object\ndef chosen(): pass\n"
    )
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


@pytest.mark.parametrize("corruption", ("node", "binding", "execution", "owner"))
def test_equal_foreign_receipts_cannot_authenticate_source_function(corruption):
    environment = execution("def chosen(): pass\n")
    node, context, binding = function(environment)
    if corruption == "node":
        result = SourceCreatedFunctionCapture(environment, copy.deepcopy(node))
        with pytest.raises(ValueError, match="unique actual operation"):
            result.require_closed()
    elif corruption == "binding":
        with pytest.raises(ValueError, match="unique original operation"):
            environment.definition_result(context, replace(binding))
    elif corruption == "execution":
        owner = binding.target.owner
        owner.__dict__["execution"] = replace(owner.execution)
        with pytest.raises(ValueError, match="different compiler receipt"):
            environment.definition_result(context, binding)
    else:
        result = environment.definition_result(context, binding)
        with pytest.raises(ValueError, match="another source definition"):
            result.require_definition_identity(replace(binding.target.owner))


def test_class_entry_reuses_definition_owner_without_function_activation_change():
    environment = execution("class Original: pass\n")
    node = environment.module.module.body[0]
    operation = environment.definition_operation(node)
    context = environment.context_for_owner(operation.owner)
    result = environment.definition_result(context, operation.event)
    assert isinstance(result, SourceCreatedClassCapture)
    assert isinstance(result.entry, SourceDefinitionEntry)
    assert result.source_definition() == (context, operation.event)
    with pytest.raises(ValueError, match="actual function declaration"):
        SourceCreatedFunctionCapture(environment, node).require_closed()
