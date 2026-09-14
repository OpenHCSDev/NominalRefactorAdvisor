"""Native protocol controls, not claims that the converter admits these inputs.

Only test-authored fixtures execute, in disposable subprocesses. These controls
separate key compatibility and ordinary class-member installation from the
additional behavior of the real AutoRegisterMeta/ABCMeta construction protocol.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def native_control(program: str, enabled: bool) -> dict:
    """Run a bounded authored protocol example without changing analyzer state."""
    result = subprocess.run(
        [sys.executable, "-c", program, str(int(enabled))],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("abstract_member", (False, True))
def test_native_member_abstractness_is_not_class_installation(abstract_member):
    outcome = native_control(
        """
import json
import sys
from metaclass_registry import AutoRegisterMeta

abstract_member = bool(int(sys.argv[1]))
queries = []
class Marker:
    @property
    def __isabstractmethod__(self):
        queries.append(self)
        return abstract_member

class OriginalAlpha:
    tag = Marker()
class OriginalBeta:
    pass
manual = {'alpha': OriginalAlpha, 'beta': OriginalBeta}
assert not queries

registry = {}
class Root(metaclass=AutoRegisterMeta):
    __registry__ = registry
    __registry_key__ = 'registry_key'
    __skip_if_no_key__ = True
    registry_key = None
class Alpha(Root):
    registry_key = 'alpha'
    tag = Marker()
class Beta(Root):
    registry_key = 'beta'

assert type(registry) is dict and Root.__registry__ is registry
assert '__set_name__' not in vars(Marker)
assert registry['beta'] is Beta
if not abstract_member:
    assert registry['alpha'] is Alpha
assert queries and all(value is Alpha.tag for value in queries)
print(json.dumps({
    'manual_keys': sorted(manual),
    'candidate_keys': sorted(registry),
    'abstract_members': sorted(Alpha.__abstractmethods__),
    'abstractness_was_inspected': bool(queries),
}))
""",
        abstract_member,
    )
    assert outcome == {
        "manual_keys": ["alpha", "beta"],
        "candidate_keys": ["beta"] if abstract_member else ["alpha", "beta"],
        "abstract_members": ["tag"] if abstract_member else [],
        "abstractness_was_inspected": True,
    }


@pytest.mark.parametrize("mutating_handler", (False, True))
def test_native_registration_completion_does_not_prove_noninterference(
    mutating_handler,
):
    outcome = native_control(
        """
import json
import logging
import sys
from metaclass_registry import AutoRegisterMeta
from metaclass_registry.core import logger

mutating_handler = bool(int(sys.argv[1]))
registry = {}
events = []
class RegistrationHandler(logging.Handler):
    def emit(self, record):
        if record.getMessage().startswith('Auto-registered '):
            events.append(record.getMessage())
            if mutating_handler:
                registry.clear()

previous_handlers = tuple(logger.handlers)
previous_level = logger.level
previous_propagate = logger.propagate
previous_disabled = logger.disabled
handler = RegistrationHandler()
try:
    for previous in previous_handlers:
        logger.removeHandler(previous)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.disabled = False

    class OriginalAlpha:
        pass
    class OriginalBeta:
        pass
    manual = {'alpha': OriginalAlpha, 'beta': OriginalBeta}
    assert not events

    class Root(metaclass=AutoRegisterMeta):
        __registry__ = registry
        __registry_key__ = 'registry_key'
        __skip_if_no_key__ = True
        registry_key = None
    class Alpha(Root):
        registry_key = 'alpha'
    class Beta(Root):
        registry_key = 'beta'

    assert type(registry) is dict and Root.__registry__ is registry
    assert not Alpha.__abstractmethods__ and not Beta.__abstractmethods__
    if not mutating_handler:
        assert registry['alpha'] is Alpha and registry['beta'] is Beta
finally:
    logger.removeHandler(handler)
    handler.close()
    for previous in previous_handlers:
        logger.addHandler(previous)
    logger.setLevel(previous_level)
    logger.propagate = previous_propagate
    logger.disabled = previous_disabled

assert tuple(logger.handlers) == previous_handlers
assert logger.level == previous_level
assert logger.propagate == previous_propagate
assert logger.disabled == previous_disabled
print(json.dumps({
    'manual_keys': sorted(manual),
    'candidate_keys': sorted(registry),
    'declared_keys': [Alpha.registry_key, Beta.registry_key],
    'registration_events': len(events),
    'classes_concrete': True,
}))
""",
        mutating_handler,
    )
    assert outcome == {
        "manual_keys": ["alpha", "beta"],
        "candidate_keys": [] if mutating_handler else ["alpha", "beta"],
        "declared_keys": ["alpha", "beta"],
        "registration_events": 2,
        "classes_concrete": True,
    }


@pytest.mark.parametrize("module_is_key", (False, True))
def test_native_metadata_contents_depend_on_selected_policy(module_is_key):
    outcome = native_control(
        """
import json
import sys
from metaclass_registry import AutoRegisterMeta

key_attribute = '__module__' if bool(int(sys.argv[1])) else 'registry_key'
def construct(module, firstline):
    def method(self):
        raise AssertionError('Representative method must never be invoked')
    registry = {}
    root = AutoRegisterMeta('Family', (), {
        '__module__': module, '__qualname__': 'Family',
        '__registry__': registry, '__registry_key__': key_attribute,
        '__skip_if_no_key__': True, 'registry_key': None,
    })
    leaf = AutoRegisterMeta('Alpha', (root,), {
        '__module__': module, '__qualname__': 'Alpha',
        '__firstlineno__': firstline, 'registry_key': 'alpha', 'method': method,
    })
    assert not leaf.__abstractmethods__
    assert all(value is leaf for value in registry.values())
    return sorted(registry)

print(json.dumps({
    'original': construct('actual.module', 3),
    'representative': construct('inert.module', 999),
}))
""",
        module_is_key,
    )
    assert outcome == {
        "original": ["actual.module"] if module_is_key else ["alpha"],
        "representative": ["inert.module"] if module_is_key else ["alpha"],
    }


@pytest.mark.parametrize("hook_role", (False, True))
def test_fresh_function_transport_does_not_authorize_invoked_roles(hook_role):
    outcome = native_control(
        """
import json
import sys
from types import FunctionType
from metaclass_registry import AutoRegisterMeta

events = []
def member(cls):
    events.append(cls.__name__)
assert type(member) is FunctionType and vars(member) == {}
name = '__init_subclass__' if bool(int(sys.argv[1])) else 'method'
root = AutoRegisterMeta('Family', (), {name: member})
AutoRegisterMeta('Alpha', (root,), {})
print(json.dumps({'events': events}))
""",
        hook_role,
    )
    assert outcome == {"events": ["Alpha"] if hook_role else []}
