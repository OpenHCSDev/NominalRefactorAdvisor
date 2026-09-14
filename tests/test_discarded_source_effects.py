"""Discard closes evaluation and reference release, not class installation."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_namespace import ExpressionStatementEffect
from nominal_refactor_advisor.product_flow import (
    CompactEvaluatedResult,
    CompactValueDestinationKind,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("discard_effects.py"),
            "discard_effects",
            False,
            ast.parse(source),
            source,
        )
    )


@pytest.mark.parametrize("scope", ("module", "class"))
@pytest.mark.parametrize(
    "expression", ("builtins.property", '"documentation"', "(1, [2, 3])")
)
def test_native_or_literal_discard_closes_without_class_installation(scope, expression):
    statement = (
        expression if scope == "module" else "class Container:\n    " + expression
    )
    environment = _execution(
        "import builtins\n" + statement + "\nclass After:\n    pass\n"
    )
    environment.require_class_creation(environment.module.module.body[-1])


@pytest.mark.parametrize(
    "expression",
    ("Value()", "(held := Value())", "dict(vars(builtins))"),
)
def test_unproved_result_release_remains_open(expression):
    environment = _execution(
        "import builtins\n"
        "class Value:\n"
        "    def __del__(self):\n"
        "        builtins.property = builtins.object\n"
        + expression
        + "\nclass After:\n    pass\n"
    )
    with pytest.raises(ValueError):
        environment.require_class_creation(environment.module.module.body[-1])


def test_discard_uses_actual_completion_event_and_rejects_equal_foreign_ast():
    source = "import builtins\nbuiltins.property\n"
    environment = _execution(source)
    node = environment.module.module.body[1]
    site = next(site for site in environment.effects.sites if site.trigger is node)
    (effect,) = site.effects
    assert isinstance(effect, ExpressionStatementEffect)
    (operation,) = effect.application_operations(site, environment.source)
    assert operation.node is node
    assert isinstance(operation.event, CompactEvaluatedResult)
    assert operation.event.destination.use is CompactValueDestinationKind.DISCARDED
    assert operation.event.value_use.position.dominates(operation.event.position)
    environment.require_discard(node)
    foreign = ast.parse(source).body[1]
    assert ast.dump(foreign, include_attributes=True) == ast.dump(
        node, include_attributes=True
    )
    with pytest.raises(ValueError, match="unique original evaluated result"):
        environment.require_discard(foreign)


def test_native_discard_retention_and_installation_are_distinct():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import weakref
events = []
references = []
class Value:
    def __del__(self):
        events.append("released")
    def __set_name__(self, owner, name):
        events.append("installed")
def make():
    value = Value()
    references.append(weakref.ref(value, lambda unused: events.append("weakref")))
    return value
make()
assert events == ["released", "weakref"]
events.clear()
(held := make())
held
assert events == []
class Container:
    held
assert events == []
class Installed:
    member = held
assert events == ["installed"]
events.clear()
del held
assert events == []
del Installed.member
assert events == ["released", "weakref"]
""",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
