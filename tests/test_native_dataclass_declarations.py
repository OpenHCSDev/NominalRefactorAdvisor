"""Dataclass discovery and native identity derive from the same declaration."""

import dataclasses
import json
import pickle
import subprocess
import sys

import pytest

from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.class_index import (
    DataclassRuntimeDeclaration as ClassIndexDataclassDeclaration,
)
from nominal_refactor_advisor.native_declarations import DataclassRuntimeDeclaration


@pytest.mark.parametrize(
    "member,declaration",
    (
        (DataclassRuntimeDeclaration.DATACLASS, dataclasses.dataclass),
        (DataclassRuntimeDeclaration.FIELD, dataclasses.field),
    ),
)
def test_discovery_labels_and_runtime_expectations_share_the_native_owner(
    member, declaration
):
    assert ClassIndexDataclassDeclaration is DataclassRuntimeDeclaration
    assert member.declaration is declaration
    native = member.native_declaration
    assert native is member.native_declaration
    assert native.declaration is declaration
    assert member.qualified_name == native.qualified_name
    assert member.value == declaration.__name__
    assert (
        DataclassRuntimeDeclaration.for_qualified_name(native.qualified_name) is member
    )
    assert CapturedNativeObject(declaration).require_native((native,)) is native
    assert json.loads(json.dumps(member)) == declaration.__name__
    assert pickle.loads(pickle.dumps(member)) is member


@pytest.mark.parametrize("member", tuple(DataclassRuntimeDeclaration))
def test_matching_metadata_is_discovery_not_native_identity(member):
    def impostor(*args, **kwargs):
        raise AssertionError("The analyzer must not call the lookalike")

    impostor.__name__ = member.declaration.__name__
    impostor.__qualname__ = member.declaration.__qualname__
    impostor.__module__ = member.declaration.__module__
    with pytest.raises(ValueError, match="required native declaration"):
        CapturedNativeObject(impostor).require_native((member.native_declaration,))


def test_native_owner_can_be_imported_and_used_before_class_index():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dataclasses\n"
            "from nominal_refactor_advisor.native_declarations import DataclassRuntimeDeclaration as Owner\n"
            "assert Owner.DATACLASS.native_declaration.declaration is dataclasses.dataclass\n"
            "from nominal_refactor_advisor.class_index import DataclassRuntimeDeclaration as Imported\n"
            "assert Imported is Owner\n",
        ],
        check=True,
        timeout=10,
    )
