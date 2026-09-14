"""Compiler module setup is execution evidence, not a supplied entry binding."""

from dataclasses import replace
import subprocess
import sys

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceViolation,
    NamespaceMemberInventory,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import (
    SourceModuleAnnotationNamespace,
    SourceModuleExecution,
)
from test_source_function_result import execution


def result_at_read(environment):
    node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    result = environment.capture_value(node)
    result.require_closed()
    return result, environment.required_prefix(read.context, read.use.position)


@pytest.mark.parametrize("future", (False, True))
def test_original_module_setup_supplies_both_read_and_complete_membership(future):
    if not future and sys.version_info >= (3, 14):
        pytest.skip("Deferred annotations have no immediate annotation dictionary")
    source = (
        "from __future__ import annotations\n" if future else ""
    ) + "value: int = 1\nresult = __annotations__\n"
    expected = "'int'" if future else "int"
    subprocess.run(
        [sys.executable, "-c", source + f"assert result == {{'value': {expected}}}\n"],
        check=True,
        timeout=10,
    )
    environment = execution(source)
    original_entries = dict(environment.entry.initial_entries)
    result, prefix = result_at_read(environment)
    assert isinstance(result, SourceModuleAnnotationNamespace)
    assert result is environment.initial_contents.annotation_namespace
    contents = NamespaceMemberInventory(environment.kernel, result, prefix)
    assert contents.names == {"value"}
    value = contents.require_member("value")
    if future:
        assert value.require_native_scalar() == "int"
    else:
        value.require_native_identity(NativeDeclaration(int))
    assert (
        "__annotations__"
        in NamespaceMemberInventory(environment.kernel, environment.entry, prefix).names
    )
    assert dict(environment.entry.initial_entries) == original_entries
    assert "__annotations__" not in original_entries


def test_compiler_setup_precedes_an_unreached_annotated_statement():
    source = "from __future__ import annotations\nresult = __annotations__\nif False:\n    value: int\n"
    subprocess.run(
        [sys.executable, "-c", source + "assert result == {}\n"], check=True, timeout=10
    )
    environment = execution(source)
    node = environment.module.module.body[1].value
    result = environment.capture_value(node)
    result.require_closed()
    assert isinstance(result, SourceModuleAnnotationNamespace)
    assert environment.entry.member("__annotations__") is None


@pytest.mark.parametrize("initial_value", (None, {"retained": 1}))
def test_preexisting_module_annotation_value_is_preserved_not_replaced(initial_value):
    source = "from __future__ import annotations\nresult = __annotations__\nif False:\n    value: int\n"
    environment = execution(source)
    captured = CapturedNativeObject(initial_value)
    entry = replace(
        environment.entry,
        bindings={**environment.entry.initial_entries, "__annotations__": captured},
    )
    environment = SourceModuleExecution(entry)
    actual = environment.capture_value(environment.module.module.body[1].value)
    assert actual is captured
    with pytest.raises(ValueError, match="initial absence"):
        environment.initial_contents.annotation_namespace.require_closed()


def test_unknown_existing_annotation_value_is_not_converted_to_a_fresh_dictionary():
    environment = execution(
        "from __future__ import annotations\nvalue: int\nresult = __annotations__\n"
    )
    unknown = OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
    entry = replace(
        environment.entry,
        bindings={**environment.entry.initial_entries, "__annotations__": unknown},
    )
    environment = SourceModuleExecution(entry)
    assert environment.initial_contents.member("__annotations__") is unknown
    with pytest.raises(ValueError):
        result_at_read(environment)


def test_source_reassignment_uses_normal_storage_instead_of_replaying_setup():
    source = "from __future__ import annotations\n__annotations__ = {}\nvalue: int\nresult = __annotations__\n"
    subprocess.run(
        [sys.executable, "-c", source + "assert result == {'value': 'int'}\n"],
        check=True,
        timeout=10,
    )
    environment = execution(source)
    result, prefix = result_at_read(environment)
    assert result is not environment.initial_contents.annotation_namespace
    assert NamespaceMemberInventory(environment.kernel, result, prefix).names == {
        "value"
    }


def test_module_storage_cannot_be_reused_in_a_different_execution_or_kernel():
    source = (
        "from __future__ import annotations\nvalue: int\nresult = __annotations__\n"
    )
    first, other = execution(source), execution(source)
    value, prefix = result_at_read(first)
    other_value, other_prefix = result_at_read(other)
    assert value is not other_value
    with pytest.raises(ValueError):
        replace(value).require_closed()
    with pytest.raises(ValueError):
        value.require_available(other.kernel, other_prefix)
    with pytest.raises(ValueError):
        first.entry_contents(other.kernel, first.entry, prefix)


def test_a_class_setup_is_not_misidentified_as_module_setup():
    environment = execution(
        "from __future__ import annotations\nclass Inner:\n    value: int\nresult = Inner\n"
    )
    assert environment.module.native_compilation.module_annotation_setup is None
    assert "__annotations__" not in environment.initial_contents.names
    result_at_read(environment)


@pytest.mark.parametrize("stored", (None, "1", "'text'"))
def test_dictionary_release_uses_current_contents_for_ordinary_source_dictionaries(
    stored,
):
    source = "values = {}\n"
    if stored is not None:
        source += f"values['item'] = {stored}\n"
    source += "values = {}\nresult = values\n"
    subprocess.run(
        [sys.executable, "-c", source + "assert result == {}\n"], check=True, timeout=10
    )
    environment = execution(source)
    result, prefix = result_at_read(environment)
    assert NamespaceMemberInventory(environment.kernel, result, prefix).names == set()


def test_dictionary_release_does_not_use_empty_birth_after_storing_a_finalizer():
    source = (
        "class Payload:\n"
        "    def __del__(self): print('released')\n"
        "values = {}\nvalues['item'] = Payload()\nvalues = {}\nresult = values\n"
    )
    actual = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert actual.stdout.strip() == "released"
    with pytest.raises(ValueError):
        result_at_read(execution(source))
