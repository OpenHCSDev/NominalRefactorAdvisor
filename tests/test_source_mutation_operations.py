"""One original mutation join, independent of storage interpretation."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactMutation,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def project(text):
    return source_product_flow_projection(
        ParsedModule(Path("mutations.py"), "mutations", False, ast.parse(text), text)
    )


@pytest.mark.parametrize(
    "text",
    (
        "value = 1",
        "receiver.attribute = 1",
        "receiver['key'] = 1",
        "def method(): pass",
        "class Owner: pass",
        "import builtins",
    ),
)
def test_mutation_join_retains_original_operation_event_and_context(text):
    source = project(text)
    operation = next(
        operation
        for operation in source.operations
        if isinstance(operation.event, CompactMutation)
    )
    assert source.mutation_operation(operation.node) is operation
    context = source.context_for_owner(operation.owner)
    assert source.source_operation(context, operation.event) is operation
    assert source.mutation_operation(operation.node).event is operation.event


@pytest.mark.parametrize("corruption", ("event", "owner", "duplicate"))
def test_mutation_join_rejects_foreign_or_ambiguous_canonical_registration(corruption):
    source = project("receiver['key'] = 1")
    operation = next(
        operation
        for operation in source.operations
        if isinstance(operation.event, CompactMutation)
    )
    if corruption == "duplicate":
        operations = (*source.operations, operation)
    else:
        replacement = (
            replace(operation, event=replace(operation.event))
            if corruption == "event"
            else replace(operation, owner=replace(operation.owner))
        )
        operations = tuple(
            replacement if site is operation else site for site in source.operations
        )
    corrupted = replace(source, operations=operations)
    with pytest.raises(ValueError):
        corrupted.mutation_operation(operation.node)


def test_join_rejects_foreign_equal_span_node_and_nonmutation_expression():
    source = project("value = 1")
    foreign = ast.parse("value = 1").body[0].targets[0]
    for node in (foreign, source.module.module.body[0].value):
        with pytest.raises(ValueError):
            source.mutation_operation(node)


def test_multi_binding_import_is_not_silently_narrowed_to_one_mutation():
    source = project("import builtins, math")
    with pytest.raises(ValueError, match="unique actual operation"):
        source.mutation_operation(source.module.module.body[0])


def test_definition_consumer_keeps_target_obligation_after_common_join():
    source = project("value = 1")
    environment = SourceModuleExecution.from_module(source.module)
    with pytest.raises(ValueError, match="definition binding"):
        environment.definition_operation(source.module.module.body[0].targets[0])


def test_storage_consumers_keep_target_obligations_after_common_join():
    source = project("value = 1")
    environment = SourceModuleExecution.from_module(source.module)
    node = source.module.module.body[0].targets[0]
    with pytest.raises(ValueError, match="attribute target"):
        environment.require_namespace_write(node)
    with pytest.raises(ValueError, match="evaluated item assignment"):
        environment.require_item_write(node)
