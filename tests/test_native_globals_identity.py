"""Native globals borrows the admitted frame's namespace, never a name inventory."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CreatedNamespaceDictionary,
    InitialNativeIsland,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_call import (
    CopiedNativeNamespace,
    NativeDictCopyCall,
    NativeDictionaryResultCall,
    NativeGlobalsCall,
    NativeVarsCall,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("globals_owner.py"), "globals_owner", False, ast.parse(source), source
        )
    )


def test_created_dictionary_behavior_has_one_inherited_owner():
    assert (
        SourceModuleEntryPremise.dictionary_namespace
        is CopiedNativeNamespace.dictionary_namespace
        is CreatedNamespaceDictionary.dictionary_namespace
    )
    assert (
        SourceModuleEntryPremise.item_write_effect
        is CopiedNativeNamespace.item_write_effect
        is CreatedNamespaceDictionary.item_write_effect
    )
    assert (
        SourceModuleEntryPremise.require_release_in
        is CopiedNativeNamespace.require_release_in
        is CreatedNamespaceDictionary.require_release_in
    )


@pytest.mark.parametrize(
    "protocol", (NativeGlobalsCall, NativeVarsCall, NativeDictCopyCall)
)
def test_namespace_admission_is_inherited_from_the_call_authority(protocol):
    assert protocol.namespace is NativeDictionaryResultCall.namespace


@pytest.mark.parametrize("protocol", (NativeGlobalsCall, NativeVarsCall))
def test_borrowed_namespace_closure_uses_the_native_call_contract(protocol):
    assert protocol.require_closed is NativeDictionaryResultCall.require_closed


def test_copy_closure_refines_call_completion_with_original_storage_admission():
    assert (
        NativeDictCopyCall.require_closed is CreatedNamespaceDictionary.require_closed
    )


@pytest.mark.parametrize(
    "source",
    (
        "result = globals()\n",
        "borrowed = globals()\nalias = borrowed\nresult = alias\n",
        "kept = globals\nglobals = object\nresult = kept()\n",
        "borrowed = globals()\nborrowed = property\nresult = globals()\n",
        "import builtins\nresult = builtins.globals()\n",
    ),
)
def test_globals_returns_the_original_entry_not_a_reconstructed_dictionary(source):
    environment = execution(source)
    node = environment.module.module.body[-1].value
    result = environment.capture_value(node)
    assert result is environment.entry
    assert not isinstance(result, CapturedNativeObject)
    assert result.dictionary_namespace(environment.initial) is environment.entry
    assert result.as_builtin_namespace(environment.initial) is environment.entry
    assert "result" not in environment.entry.initial_entries
    assert environment.capture_value(node) is result


def test_class_body_globals_is_module_namespace_not_class_locals():
    environment = execution("class Body:\n    observed = globals()\n")
    owner = environment.module.module.body[0]
    node = owner.body[0].value
    result = environment.capture_value(node)
    assert result is environment.entry
    context, call = environment.source_call(node)
    frame = environment.required_prefix(context, call.position).endpoint.frame
    assert frame.globals is result
    assert frame.locals is not result
    environment.require_class_creation(owner)


def test_copy_of_globals_uses_positioned_bindings_not_initial_entries_or_later_state():
    environment = execution(
        "selected = property\ncopied = dict(globals())\n"
        "selected = object\nresult = copied\n"
    )
    result = environment.capture_value(environment.module.module.body[-1].value)
    assert isinstance(result, CopiedNativeNamespace)
    assert result.parent is environment.entry
    assert environment.entry.member("selected") is None
    assert (
        result.member("selected")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_borrowed_namespace_rejects_a_foreign_admission():
    environment = execution("result = globals()\n")
    result = environment.capture_value(environment.module.module.body[-1].value)
    assert result is environment.entry
    with pytest.raises(ValueError, match="different entry premise"):
        result.dictionary_namespace(InitialNativeIsland(()))


def test_globals_borrows_now_but_dict_copies_after_keyword_evaluation():
    environment = execution(
        "selected = property\n"
        "copied = dict(globals(), earlier=selected, change=(selected := object))\n"
        "result = copied\n"
    )
    result = environment.capture_value(environment.module.module.body[-1].value)
    assert isinstance(result, CopiedNativeNamespace)
    assert result.parent is environment.entry
    assert (
        result.member("selected")
        .require_native_identity(NativeDeclaration(object))
        .declaration
        is object
    )
    assert (
        result.member("earlier")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_source_function_body_requires_its_actual_activation():
    environment = execution("def body():\n    return globals()\n")
    node = environment.module.module.body[0].body[0].value
    assert isinstance(environment.capture_value(node), OpenCapturedReference)


@pytest.mark.parametrize(
    "source",
    (
        "result = globals(1)\n",
        "result = globals(value=1)\n",
        "result = globals(*[])\n",
        "result = globals(**{})\n",
        "def globals(): return {}\nresult = globals()\n",
        "unknown()\nresult = globals()\n",
    ),
)
def test_unproved_call_or_namespace_effect_is_not_admitted(source):
    environment = execution(source)
    result = environment.capture_value(environment.module.module.body[-1].value)
    assert isinstance(result, OpenCapturedReference)


def test_native_globals_item_installation_supplies_the_actual_bound_value():
    source = "globals()['property'] = object\nresult = property\n"
    environment = execution(source)
    environment.require_item_write(environment.module.module.body[0].targets[0])
    result = environment.capture_value(environment.module.module.body[-1].value)
    result.require_native_identity(NativeDeclaration(object))
    native = subprocess.run(
        [sys.executable, "-c", source + "assert result is object\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


def test_native_namespace_controls_in_a_fresh_process():
    source = """
selected = property
borrowed = globals()
assert borrowed is globals()
copied = dict(borrowed)
selected = object
assert copied['selected'] is property
assert borrowed['selected'] is object
selected = property
copied = dict(globals(), earlier=selected, change=(selected := object))
assert copied['selected'] is object
assert copied['earlier'] is property
class Body:
    observed = globals()
assert Body.observed is borrowed
kept = globals
globals = object
assert kept() is borrowed
"""
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr
