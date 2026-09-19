"""The standard entry admits dataclass imports, not decorator execution effects."""

import dataclasses
import sys

import pytest
from test_product_flow_authority import _module
from test_source_function_result import execution

from nominal_refactor_advisor.native_declarations import (
    DataclassRuntimeDeclaration,
)
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise


@pytest.mark.parametrize(
    "statement,expression",
    (
        ("from dataclasses import dataclass", "dataclass"),
        ("from dataclasses import dataclass as decorate", "decorate"),
        ("import dataclasses as dc", "dc.dataclass"),
    ),
)
def test_default_entry_retains_actual_import_identity(statement, expression):
    environment = execution(f"{statement}\nheld = {expression}\n")
    capture = environment.capture_value(environment.module.module.body[-1].value)
    native = DataclassRuntimeDeclaration.DATACLASS.native_declaration
    assert capture.require_native((native,)) is native
    assert not environment.entry.operation_conditions


def test_default_import_proves_binding_without_factory_execution():
    environment = execution("from dataclasses import dataclass\nheld = dataclass()\n")
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    authority = environment.call_authority(context, call)
    calls = []
    native_code = dataclasses.dataclass.__code__

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is native_code:
            calls.append(frame)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        authority.require_closed()
    finally:
        sys.setprofile(previous)
    assert not calls
    assert not environment.entry.operation_conditions


def test_original_spelling_does_not_restore_a_rebound_native_declaration():
    environment = execution(
        "from dataclasses import dataclass\n"
        "dataclass = object\n"
        "held = dataclass\n"
    )
    capture = environment.capture_value(environment.module.module.body[-1].value)
    with pytest.raises(ValueError):
        capture.require_native(
            (DataclassRuntimeDeclaration.DATACLASS.native_declaration,)
        )


def test_changed_library_member_is_not_the_original_declared_function(monkeypatch):
    calls = []

    def changed(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("The analyzer must not execute a library replacement")

    monkeypatch.setattr(dataclasses, "dataclass", changed)
    environment = execution("from dataclasses import dataclass\nheld = dataclass\n")
    capture = environment.capture_value(environment.module.module.body[-1].value)
    with pytest.raises(ValueError):
        capture.require_native(
            (DataclassRuntimeDeclaration.DATACLASS.native_declaration,)
        )
    assert not calls


def test_native_module_registration_cannot_be_replaced_by_same_named_source():
    source = source_product_flow_projection(_module("dataclasses", "value = 1\n"))
    with pytest.raises(ValueError, match="replace an admitted native module"):
        ImportedSourceModuleEntryPremise.from_source(source)


def test_unlisted_native_import_remains_unadmitted():
    environment = execution("import os\nheld = os\n")
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[-1].value
        ).require_closed()
