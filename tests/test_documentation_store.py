"""Documentation uses its original owner and ordinary destination storage."""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactEvaluatedAssignment,
    CompactEvaluatedResult,
    CompactMutation,
    CompactValueDestinationKind,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("documentation.py"), "documentation", False, ast.parse(source), source
        )
    )


@pytest.mark.parametrize(
    "source,stored",
    (
        ('"module documentation"\n', True),
        ('class Owner:\n    "class documentation"\n', True),
        ('def owner():\n    "function documentation"\n', False),
        ('async def owner():\n    "function documentation"\n', False),
        ('class Owner:\n    pass\n    "ordinary expression"\n', False),
        ('class Owner:\n    if True:\n        "branch expression"\n', False),
    ),
)
def test_only_original_namespace_documentation_has_a_store(source, stored):
    env = execution(source)
    node = next(
        node for node in ast.walk(env.module.module) if isinstance(node, ast.Expr)
    )
    operations = env.source.operations_by_node[node]
    results = [
        operation
        for operation in operations
        if isinstance(operation.event, CompactEvaluatedResult)
    ]
    assert len(results) == 1
    result = results[0]
    mutations = [
        operation
        for operation in operations
        if isinstance(operation.event, CompactMutation)
    ]
    assert bool(mutations) is stored
    if stored:
        assert len(mutations) == 1
        mutation = mutations[0]
        assert isinstance(mutation.event, CompactEvaluatedAssignment)
        assert mutation.event.result is result.event
        assert mutation.event.target.bound_name == "__doc__"
        assert result.event.value_use.position.dominates(result.position)
        assert result.position.dominates(mutation.position)
        site = next(site for site in env.effects.sites if site.trigger is node)
        assert len(site.effects) == 1
        assert site.effects[0].application_operations(site, env.source) == (mutation,)
        assert mutation.node is node
        with pytest.raises(ValueError, match="original"):
            env._require_binding_operation(replace(mutation))
    else:
        assert result.event.destination.use is CompactValueDestinationKind.DISCARDED


@pytest.mark.parametrize(
    "body",
    (
        '"initial"\n    __doc__ = "final"',
        '"initial"\n    del __doc__',
        '"global documentation"\n    global __doc__',
    ),
)
def test_documentation_uses_normal_replacement_deletion_and_global_routing(body):
    source = "class Owner:\n    " + body + "\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert native.returncode == 0, native.stderr
    env = execution(source)
    env.require_class_creation(env.module.module.body[0])


def test_module_documentation_can_precede_a_class():
    env = execution('"module documentation"\nclass Owner: pass\n')
    env.require_class_creation(env.module.module.body[-1])


def test_class_documentation_is_visible_to_later_body_reads():
    env = execution('class Owner:\n    "original documentation"\n    saved = __doc__\n')
    owner = env.module.module.body[0]
    env.require_class_creation(owner)
    assert (
        env.class_entry(owner).completion_member("saved").require_native_text()
        == "original documentation"
    )


def test_invalid_initial_documentation_overwritten_before_construction():
    source = "class Owner:\n    '\\ud800'\n    __doc__ = 'valid'\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    if native.returncode:
        assert "UnicodeEncodeError" in native.stderr
        with pytest.raises(ValueError):
            env = execution(source)
            env.require_class_creation(env.module.module.body[0])
    else:
        env = execution(source)
        env.require_class_creation(env.module.module.body[0])
