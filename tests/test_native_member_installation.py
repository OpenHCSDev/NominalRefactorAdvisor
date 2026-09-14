"""Member hook evidence follows the captured value's actual runtime type."""

import ast
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise


@pytest.mark.parametrize("value", (object, property, type, 1, None, {}, [], ()))
def test_static_native_types_without_member_hooks_are_admitted(value):
    CapturedNativeObject(value).require_class_installation()


def test_descriptor_instance_is_not_confused_with_its_class():
    with pytest.raises(ValueError, match="installation hook"):
        CapturedNativeObject(property()).require_class_installation()


def test_unproved_capture_keeps_its_original_rejection():
    value = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    with pytest.raises(ValueError, match="unproved_binding"):
        value.require_class_installation()


def test_heap_type_hook_lookup_is_not_executed():
    observed = []

    class Descriptor:
        def __getattribute__(self, name):
            observed.append(name)
            raise AssertionError("Analyzer executed a target lookup")

        def __set_name__(self, owner, name):
            observed.append((owner, name))

    with pytest.raises(ValueError, match="static type"):
        CapturedNativeObject(Descriptor()).require_class_installation()
    assert not observed


@pytest.mark.parametrize(
    "source",
    (
        "class Owner:\n    member = property\n",
        "Alias = object\nclass Owner:\n    member = Alias\n",
        "class Owner:\n    first = globals()\n    second = first\n",
    ),
)
def test_class_members_use_captured_value_protocols(source):
    module = ParsedModule(Path("member.py"), "member", False, ast.parse(source), source)
    execution = SourceModuleExecution.from_module(module)
    execution.require_class_creation(module.module.body[-1])
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr


def test_native_creation_calls_instance_hook_but_not_the_class_member():
    source = """
calls = []
class Descriptor:
    def __set_name__(self, owner, name):
        calls.append((owner.__name__, name))
class ClassMember:
    member = Descriptor
class InstanceMember:
    member = Descriptor()
print(calls)
"""
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert ast.literal_eval(native.stdout) == [("InstanceMember", "member")]


def test_member_hook_absence_does_not_prove_truth_evaluation():
    calls = []

    class Condition:
        def __bool__(self):
            calls.append("truth")
            return True

    target = Condition()
    proxy = weakref.proxy(target)
    # This native proxy type has no member installation hook. Its truth
    # operation still forwards into source behavior and is a different proof.
    CapturedNativeObject(proxy).require_class_installation()
    source = "class Owner:\n    if flag:\n        pass\n"
    module = ParsedModule(
        Path("condition.py"), "condition", False, ast.parse(source), source
    )
    seed = SourceModuleExecution.from_module(module)
    entry = SourceModuleEntryPremise(
        seed.source,
        seed.initial,
        dict(seed.entry.initial_entries, flag=CapturedNativeObject(proxy)),
        seed.entry.builtins,
    )
    with pytest.raises(ValueError):
        SourceModuleExecution(entry).require_class_creation(module.module.body[-1])
    assert not calls
    exec(source, {"flag": proxy})
    assert calls == ["truth"]
