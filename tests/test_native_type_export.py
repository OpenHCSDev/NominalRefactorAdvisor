"""Portable native identity comes from actual exports, not advertised names."""

import pickle
import types

import pytest

from nominal_refactor_advisor.native_declarations import NativeTypeDeclaration


@pytest.mark.parametrize(
    "native_type",
    (
        int,
        str,
        tuple,
        dict,
        type(None),
        types.CodeType,
        types.FunctionType,
        types.CellType,
    ),
)
def test_native_type_roundtrip_uses_actual_export_identity(native_type):
    declaration = NativeTypeDeclaration(native_type)
    restored = pickle.loads(pickle.dumps(declaration))
    assert restored.declaration is native_type
    assert restored == declaration
    assert restored.qualified_name == declaration.qualified_name


def test_arbitrary_source_type_does_not_acquire_a_native_export():
    class SourceType:
        pass

    with pytest.raises(ValueError, match="standard-library export"):
        pickle.dumps(NativeTypeDeclaration(SourceType))


def test_non_type_export_is_not_native_type_evidence():
    with pytest.raises(ValueError, match="exact type"):
        NativeTypeDeclaration.from_qualified_name("builtins.len")
