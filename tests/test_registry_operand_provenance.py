"""Registration evidence retains actual operands for later binding proof."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.manual_registry import (
    AutoRegisterInstanceViewComponent,
    DirectManualRegistryComponent,
    DirectModuleClassGraph,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def parsed(source):
    return ParsedModule(
        Path("/repo/operands.py"), "operands", False, ast.parse(source), source
    )


@pytest.mark.parametrize(
    "registrations",
    (
        "REGISTRY = {'alpha': Alpha, 'beta': Beta}\n",
        "REGISTRY = {}\nREGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\n",
    ),
)
def test_manual_entries_retain_original_values_and_class_declarations(registrations):
    module = parsed("class Alpha: pass\nclass Beta: pass\n" + registrations)
    component = DirectManualRegistryComponent.from_module_anchor(module.module, "Alpha")
    operands = tuple(
        node
        for node in ast.walk(module.module)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in ("Alpha", "Beta")
    )
    for entry, original in zip(component.entries, operands, strict=True):
        assert entry.value_node is original
        assert (
            entry.class_node is module.module.body[("Alpha", "Beta").index(original.id)]
        )


def test_dictionary_entry_can_use_existing_original_binding_proof():
    module = parsed(
        "class Alpha: pass\nclass Beta: pass\nREGISTRY = {'alpha': Alpha, 'beta': Beta}\n"
    )
    component = DirectManualRegistryComponent.from_module_anchor(module.module, "Alpha")
    execution = SourceModuleExecution.from_module(module)
    for entry in component.entries:
        definition = execution.definition_operation(entry.class_node)
        execution.capture(entry.value_node).require_definition_identity(
            definition.event.target.owner
        )


def test_lexical_class_name_does_not_prove_a_later_registration_value():
    module = parsed(
        "class Alpha: pass\nclass Beta: pass\nAlpha = Beta\nREGISTRY = {'alpha': Alpha, 'beta': Beta}\n"
    )
    component = DirectManualRegistryComponent.from_module_anchor(module.module, "Alpha")
    entry = component.entries[0]
    assert entry.class_node is module.module.body[0]
    assert entry.value_node is module.module.body[-1].value.values[0]
    execution = SourceModuleExecution.from_module(module)
    original_definition = execution.definition_operation(entry.class_node)
    with pytest.raises(ValueError):
        execution.capture(entry.value_node).require_definition_identity(
            original_definition.event.target.owner
        )


def test_instance_entry_retains_whole_original_call_and_callee():
    module = parsed(
        "class Root: pass\nclass Alpha(Root): pass\nclass Beta(Root): pass\nREGISTRY = {'alpha': Alpha(), 'beta': Beta()}\n"
    )
    tree = module.module
    component = AutoRegisterInstanceViewComponent.from_assignment(
        tree, tree.body[0], tree.body[-1], DirectModuleClassGraph(tree)
    )
    assert component is not None
    for entry, call in zip(component.entries, tree.body[-1].value.values, strict=True):
        assert entry.value_node is call
        assert isinstance(entry.value_node, ast.Call)
        assert entry.value_node.func is call.func


@pytest.mark.parametrize(
    "expression",
    (
        "{'alpha': Alpha, **other}",
        "{'alpha': Alpha, 'other': unknown}",
    ),
)
def test_partial_dictionary_does_not_become_a_complete_entry_set(expression):
    module = parsed("class Alpha: pass\nREGISTRY = " + expression + "\n")
    assert (
        DirectManualRegistryComponent.direct_entries(
            module.module, DirectModuleClassGraph(module.module).classes_by_name
        )
        == ()
    )
