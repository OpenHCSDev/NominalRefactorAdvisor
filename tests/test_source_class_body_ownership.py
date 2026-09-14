"""Prepared-body evidence shares ownership without admitting other constructors."""

import ast
from dataclasses import fields
from inspect import isabstract
from typing import get_type_hints

import pytest

from nominal_refactor_advisor.captured_reference import NamespaceMemberInventory
from nominal_refactor_advisor.source_execution import (
    SourceClassBodyEntryABC,
    SourceClassEntry,
    SourceCreatedClassCapture,
    SourceExecutionKernel,
    SourceFreshCellCapture,
    SourceModuleExecution,
    SourceNativeNamespaceABC,
    SourceNativeStorageABC,
    SourceNativeTypeCapture,
)
from test_documentation_store import execution
from test_registry_original_values import authored_runtime


def test_body_contract_requires_capture_and_preparation_before_instantiation():
    assert isabstract(SourceClassBodyEntryABC)
    assert not isabstract(SourceClassEntry)
    assert SourceClassBodyEntryABC.__abstractmethods__ == frozenset(
        (
            "_exact_class_capture_resolution",
            "require_native_preparation",
            "construction_admission",
            "_created_result",
        )
    )
    env = execution("class Sample: pass\n")
    with pytest.raises(TypeError, match="abstract"):
        SourceClassBodyEntryABC(env, env.module.module.body[0])


def test_canonical_class_entry_inherits_body_without_a_second_namespace_owner():
    source = "class Sample:\n    selected = object\ntail = Sample\n"
    env = execution(source)
    node = env.module.module.body[0]
    entry = env.class_entry(node)
    assert type(entry) is SourceClassEntry
    assert isinstance(entry, SourceClassBodyEntryABC)
    assert tuple(field.name for field in fields(entry)) == ("execution", "node")
    assert env.class_entry(node) is entry
    env.require_class_creation(node)
    assert entry.frame.locals is entry
    assert entry.frame is entry.frame
    contents = NamespaceMemberInventory(env.kernel, entry, entry.completion_prefix)
    assert contents.namespace is entry
    selected = contents.require_member("selected")
    assert selected is entry.completion_member("selected")
    assert selected.proves_same_object(env.capture(node.body[0].value))
    runtime = authored_runtime(source)
    assert runtime["Sample"].selected is object
    assert runtime["tail"] is runtime["Sample"]
    capture = env.capture_value(env.module.module.body[-1].value)
    assert isinstance(capture, SourceCreatedClassCapture)
    assert capture.entry is entry
    assert capture.native_type is type


def test_prologue_values_depend_on_body_not_ordinary_class_result():
    assert get_type_hints(SourceNativeTypeCapture)["entry"] is SourceNativeStorageABC
    assert get_type_hints(SourceFreshCellCapture)["entry"] is SourceNativeNamespaceABC
    assert (
        get_type_hints(SourceNativeNamespaceABC.native_class_entry.fget)["return"]
        is SourceClassBodyEntryABC
    )
    assert "native_value" in vars(SourceNativeStorageABC)
    assert "native_value" not in vars(SourceClassBodyEntryABC)
    for name in (
        "prefix",
        "capture_operation",
        "completion_member",
        "require_installed_result",
        "result",
    ):
        assert name not in vars(SourceClassEntry)
        assert name in vars(SourceClassBodyEntryABC)
    for name in ("mro_type",):
        assert name in vars(SourceClassEntry)
        assert name not in vars(SourceClassBodyEntryABC)


def test_generic_class_lookup_uses_the_canonical_entry_result_contract():
    assert (
        get_type_hints(SourceModuleExecution.class_entry)["return"]
        is SourceClassBodyEntryABC
    )
    names = SourceExecutionKernel._selected_class_resolution.__code__.co_names
    assert "result" in names
    assert "SourceCreatedClassCapture" not in names
    assert SourceClassBodyEntryABC._created_result.__isabstractmethod__
    assert (
        SourceClassEntry._created_result is not SourceClassBodyEntryABC._created_result
    )


@pytest.mark.parametrize(
    "source",
    (
        "class Sample: pass\ntail = Sample\n",
        "class Base: pass\nclass Sample(Base): pass\ntail = Sample\n",
        "class Sample: pass\nalias = Sample\nclass Other: pass\nSample = Other\ntail = alias\n",
    ),
)
def test_entry_result_retains_actual_creation_after_inheritance_or_rebinding(source):
    env = execution(source)
    node = next(
        node
        for node in env.module.module.body
        if isinstance(node, ast.ClassDef) and node.name == "Sample"
    )
    entry = env.class_entry(node)
    result = entry.result()
    assert type(result) is SourceCreatedClassCapture
    assert result.entry is entry
    result.require_closed()
    observed = env.capture_value(env.module.module.body[-1].value)
    assert observed.proves_same_object(result)
    assert env.class_entry(node) is entry
    assert entry.frame.locals is entry
    native = authored_runtime(source)
    assert native["tail"].__name__ == "Sample"
    assert type(native["tail"]) is result.native_type is type


@pytest.mark.parametrize(
    "source",
    (
        "class Sample(metaclass=type): pass\n",
        "@unknown\nclass Sample: pass\n",
        "class Sample(unknown()): pass\n",
        "class Sample:\n    member = unknown()\n",
    ),
)
def test_result_factory_cannot_bypass_unproved_creation_or_installation(source):
    env = execution(source)
    with pytest.raises(ValueError):
        entry = env.class_entry(env.module.module.body[0])
        entry.result()


@pytest.mark.parametrize("header", ("metaclass=type", "unknown(), metaclass=type"))
def test_body_factoring_does_not_bypass_explicit_constructor_keyword_rejection(header):
    env = execution(f"class Sample({header}): pass\n")
    # The ordinary owner retains its proof boundary independently of the new
    # factory's earlier native-family selection from explicit metaclass inputs.
    entry = SourceClassEntry(env, env.module.module.body[0])
    with pytest.raises(ValueError, match="unproved construction hooks"):
        _ = entry.frame


def test_unknown_base_still_prevents_prepared_frame_admission():
    env = execution("class Sample(unknown()): pass\n")
    entry = env.class_entry(env.module.module.body[0])
    with pytest.raises(ValueError):
        _ = entry.frame
    assert "frame" not in vars(entry)


def test_body_origin_does_not_grant_class_decorator_installation():
    env = execution("@unknown\nclass Sample: pass\n")
    entry = env.class_entry(env.module.module.body[0])
    assert isinstance(entry.node, ast.ClassDef)
    with pytest.raises(ValueError):
        entry.require_installed_result()


def test_body_keeps_original_initial_island_association():
    source = "class Sample: pass\n"
    env, foreign = execution(source), execution(source)
    entry = env.class_entry(env.module.module.body[0])
    entry.require_admitted(env.initial)
    with pytest.raises(ValueError, match="foreign native admission"):
        entry.require_admitted(foreign.initial)
