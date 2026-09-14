"""Native cache callbacks are not covered by successful operation completion.

Only authored fixtures execute in disposable subprocesses. The malicious case
supplies completion, which is true, and never asserts noninterference, which is
false. These tests exercise actual Python cache callbacks, not mocked effects.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

PROGRAM = r"""
import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType
import typing

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_subscription import NativeSubscriptionAuthority
from nominal_refactor_advisor.product_flow import CompactSubscription
from nominal_refactor_advisor.source_entry import DeclaredOperationCompletion
from nominal_refactor_advisor.source_execution import SourceModuleExecution

annotation = bool(int(sys.argv[1]))
runtime_module = ModuleType('native_cache_control')
namespace = runtime_module.__dict__
events = []

class Collision:
    armed = False

    def __init__(self, target):
        self.target = target

    def __hash__(self):
        return hash(self.target)

    def __eq__(self, other):
        if self.armed:
            events.append('completing-cache-key-equality')
            sys.modules['native_cache_control'].__dict__['selected'] = int
        return self is other

number = 98765432123456789123456789
key = Collision(number)
typing.ClassVar[key]
key.armed = True

statement = (f'held: ClassVar[{number}] = None' if annotation
             else f'held = ClassVar[{number}]')
text = f'from typing import ClassVar\nselected = object\n{statement}\nobserved = selected\n'
module = ParsedModule(Path('native_cache_control.py'), 'native_cache_control',
                      False, ast.parse(text), text)
original = SourceModuleExecution.from_module(module)
operations = tuple(operation for operation in original.source.operations
                   if isinstance(operation.event, CompactSubscription))
environment = SourceModuleExecution(replace(
    original.entry,
    bindings=dict(original.entry.initial_entries),
    declared_operation_conditions=tuple(
        DeclaredOperationCompletion(operation) for operation in operations),
))

eager = not annotation or sys.version_info < (3, 14)
if not annotation:
    operation, = operations
    authority = NativeSubscriptionAuthority.for_subscription(
        environment, environment.context_for_owner(operation.owner), operation.event)
    environment.entry.require_operation_completion(authority)
    read = environment.source.value_reads_by_node[operation.node]

    def query():
        prefix = environment.required_prefix(read.context, read.use.position)
        return environment.kernel._namespace_resolution(
            environment.entry, 'selected', prefix, frozenset())
else:
    def query():
        return environment.capture_value(module.module.body[-1].value)

try:
    query().require_native_identity(NativeDeclaration(object))
except ValueError:
    refused = True
else:
    refused = False
assert refused is eager, (annotation, sys.version_info, refused)
assert not events, 'Analyzer binding inspection must not execute private cache callbacks'
assert not environment._pending

# Register the actual fresh target module only after analyzer initial admission.
sys.modules[runtime_module.__name__] = runtime_module
exec(compile(text, module.file_path, 'exec'), namespace)
assert bool(events) is eager
assert namespace['observed'] is (int if eager else object)
assert 'held' in namespace
print(json.dumps({
    'eager': eager,
    'source_capture_refused': refused,
    'actual_observed': namespace['observed'].__name__,
    'cache_callback_executed': bool(events),
    'target_completed': True,
}))
"""


@pytest.mark.parametrize("annotation", (False, True))
def test_completing_cache_callback_cannot_preserve_prior_namespace_without_behavior(
    annotation,
):
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(int(annotation))],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    eager = not annotation or sys.version_info < (3, 14)
    assert json.loads(result.stdout) == {
        "eager": eager,
        "source_capture_refused": eager,
        "actual_observed": "int" if eager else "object",
        "cache_callback_executed": eager,
        "target_completed": True,
    }
