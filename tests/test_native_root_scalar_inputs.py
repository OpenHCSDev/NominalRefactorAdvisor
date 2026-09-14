"""Actual prepared-root scalar inputs, without claiming a construction result.

Native execution is limited to test-authored fixtures in disposable subprocesses.
The default loader is never upgraded from an unknown module string to contents.
"""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NamespaceMemberInventory,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import NativeConstantValue
from nominal_refactor_advisor.source_execution import (
    SourceModuleExecution,
    SourceNativeTypeCapture,
)
from test_native_source_class_preparation import SOURCE, prepared_execution
from test_registry_native_registration_controls import native_control


def root_with_known_module(module_name):
    environment, nodes = prepared_execution()
    environment = SourceModuleExecution(
        replace(
            environment.entry,
            bindings={
                **environment.entry.initial_entries,
                "__name__": CapturedNativeObject(module_name),
            },
            declared_operation_conditions=environment.entry.operation_conditions.values(),
        )
    )
    return environment, nodes


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("__registry_key__", "registry_key"),
        ("__skip_if_no_key__", True),
        ("registry_key", None),
        ("__qualname__", "Family"),
    ),
)
def test_complete_root_exposes_owned_scalar_contents_without_a_result(name, expected):
    environment, (root, _) = prepared_execution()
    entry = environment.class_entry(root)
    prefix = entry.completion_prefix
    assert name in NamespaceMemberInventory(environment.kernel, entry, prefix).names
    actual = entry.completion_member(name).require_native_scalar()
    assert type(actual) is type(expected)
    assert actual == expected
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()
    assert not environment._pending


def test_default_module_type_fact_does_not_become_scalar_or_text_contents():
    environment, (root, _) = prepared_execution()
    value = environment.class_entry(root).completion_member("__module__")
    assert type(value) is NativeTypePremise
    assert value.native_type is str
    with pytest.raises(ValueError, match="scalar contents remain unproved"):
        value.require_native_scalar()
    with pytest.raises(ValueError, match="scalar contents remain unproved"):
        value.require_native_text()


@pytest.mark.parametrize(
    "module_name", ("supplied.actual.module", "other.actual.module")
)
def test_truthful_initial_module_contents_match_authored_native_input(module_name):
    environment, (root, leaf) = root_with_known_module(module_name)
    entry = environment.class_entry(root)
    value = entry.completion_member("__module__")
    assert type(value) is CapturedNativeObject
    assert value.require_native_scalar() == module_name
    assert value.require_native_text() == module_name

    native = native_control(
        "import json\nfrom metaclass_registry import AutoRegisterMeta\n"
        "namespace = {'Creator': AutoRegisterMeta, '__name__': "
        + repr(module_name)
        + "}\nexec("
        + repr(SOURCE)
        + ", namespace)\n"
        "assert namespace['REGISTRY'] == {'alpha': namespace['Alpha']}\n"
        "assert namespace['Family'].example.__dict__ == {}\n"
        "print(json.dumps({'module': namespace['Family'].__module__, "
        "'keys': list(namespace['REGISTRY'])}))\n",
        False,
    )
    assert native == {"module": module_name, "keys": ["alpha"]}

    # Exact inputs do not grant the absent native class result or its leaf base.
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()
    with pytest.raises(ValueError):
        environment.capture_value(leaf.bases[0]).require_closed()
    assert not environment._pending


def test_firstlineno_contents_come_from_actual_compiler_production_when_emitted():
    environment, (root, _) = prepared_execution()
    entry = environment.class_entry(root)
    names = NamespaceMemberInventory(
        environment.kernel, entry, entry.completion_prefix
    ).names
    native = native_control(
        "import json\nfrom metaclass_registry import AutoRegisterMeta\n"
        "namespace = {'Creator': AutoRegisterMeta, '__name__': 'authored'}\nexec("
        + repr(SOURCE)
        + ", namespace)\n"
        "stored = vars(namespace['Family'])\n"
        "print(json.dumps({key: stored[key] for key in ('__firstlineno__',) if key in stored}))\n",
        False,
    )
    assert ("__firstlineno__" in names) == ("__firstlineno__" in native)
    if "__firstlineno__" not in names:
        return
    value = entry.completion_member("__firstlineno__")
    assert type(value) is SourceNativeTypeCapture
    assert type(value.value) is NativeConstantValue
    actual = value.require_native_scalar()
    assert type(actual) is int
    assert actual == native["__firstlineno__"] == root.lineno
    # Equal contents on a copied producer do not acquire original source identity.
    with pytest.raises(ValueError, match="original"):
        replace(value, value=replace(value.value)).require_native_scalar()


def test_truthful_input_does_not_broaden_non_scalar_root_operands():
    environment, (root, _) = root_with_known_module("authored.module")
    entry = environment.class_entry(root)
    for name in ("__registry__", "example"):
        value = entry.completion_member(name)
        value.require_closed()
        with pytest.raises(ValueError, match="scalar contents remain unproved"):
            value.require_native_scalar()
