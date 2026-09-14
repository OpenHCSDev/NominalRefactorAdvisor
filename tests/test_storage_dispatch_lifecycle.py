"""Storage targets dispatch under one authenticated success-cache lifetime."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactMutation,
    CompactMutationResolverABC,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("storage_dispatch.py"),
            "storage_dispatch",
            False,
            ast.parse(source),
            source,
        )
    )


@pytest.mark.parametrize(
    "source, hook",
    (
        ("value = 1\n", "_binding_mutation_resolution"),
        (
            "import builtins\nbuiltins.NEW_STORAGE_DISPATCH_MEMBER = 1\n",
            "_attribute_mutation_resolution",
        ),
        ("globals()['new_storage_dispatch_member'] = 1\n", "_item_mutation_resolution"),
    ),
)
def test_one_cache_lifecycle_dispatches_each_target_once_and_authenticates_hits(
    monkeypatch, source, hook
):
    environment = execution(source)
    statement = environment.module.module.body[-1]
    operation = environment.source.mutation_operation(statement.targets[0])
    assert isinstance(environment, CompactMutationResolverABC)
    calls = []
    original = getattr(SourceModuleExecution, hook)

    def counted(self, context, mutation, *args):
        if mutation is operation.event:
            calls.append(mutation)
        return original(self, context, mutation, *args)

    monkeypatch.setattr(SourceModuleExecution, hook, counted)
    environment._require_storage_operation(operation)
    environment._require_storage_operation(operation)
    assert calls == [operation.event]
    assert operation in environment._closed_storage_operations
    # A copied operation is not a cache hit or an authorized fresh mutation.
    with pytest.raises(ValueError, match="original source operation"):
        environment._require_storage_operation(replace(operation))


@pytest.mark.parametrize(
    "source",
    (
        "value = 1\ndel missing\n",
        "import builtins\nbuiltins.NEW_STORAGE_DISPATCH_MEMBER += 1\n",
        "globals()[unknown_key()] = 1\n",
    ),
)
def test_storage_failure_is_not_published_as_completed(source):
    environment = execution(source)
    statement = environment.module.module.body[-1]
    target = (
        statement
        if isinstance(statement, ast.AugAssign)
        else statement.targets[0]
    )
    operation = environment.source.mutation_operation(target)
    for _ in range(2):
        with pytest.raises(ValueError):
            environment._require_storage_operation(operation)
        assert operation not in environment._closed_storage_operations


def test_nonmutation_cannot_enter_storage_success_cache():
    environment = execution("value = property\n")
    node = environment.module.module.body[0].value
    operation = next(
        item
        for item in environment.source.operations_by_node[node]
        if not isinstance(item.event, CompactMutation)
    )
    with pytest.raises(ValueError, match="actual mutation"):
        environment._require_storage_operation(operation)
    assert operation not in environment._closed_storage_operations
