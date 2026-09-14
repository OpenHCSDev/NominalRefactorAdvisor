"""Scalar content transport preserves native types and original source proofs."""

from dataclasses import fields, replace
import dis
import pickle
import subprocess
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import (
    NativeConstantValue,
    NativeCreationBackend,
    NativeEntryValueWindow,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativeTypedValue,
)
from nominal_refactor_advisor.native_declarations import NativeTypeDeclaration
from nominal_refactor_advisor.source_execution import SourceLiteralCapture
from test_native_text_capture import execution


@pytest.mark.parametrize("value", [None, False, True, 0, -7, 2**200, "", "axon"])
def test_actual_and_source_scalars_preserve_contents_and_exact_type(value):
    actual = CapturedNativeObject(value)
    source = f"result = {value!r}\n"
    environment = execution(source)
    captured = environment.capture_value(environment.module.module.body[0].value)
    for evidence in (actual, captured):
        result = evidence.require_native_scalar()
        assert type(result) is type(value)
        assert result == value
    assert not captured.proves_same_object(actual)


@pytest.mark.parametrize("declaration", [str, int, bool, type(None)])
def test_type_only_evidence_never_supplies_scalar_contents(declaration):
    for evidence in (
        NativeTypePremise(declaration),
        NativeTypedValue(0, (), NativeTypeDeclaration(declaration)),
    ):
        with pytest.raises(ValueError, match="scalar contents remain unproved"):
            evidence.require_native_scalar()


@pytest.mark.parametrize(
    "value", [1.25, float("nan"), 2j, b"axon", (1, 2), frozenset({1}), [], {}, Ellipsis]
)
def test_outside_scalar_domain_remains_unproved(value):
    with pytest.raises(ValueError, match="admitted exact primitive"):
        CapturedNativeObject(value).require_native_scalar()


def test_type_admission_does_not_hash_compare_or_convert_foreign_values():
    def forbidden(*args, **kwargs):
        raise AssertionError("native scalar query invoked an unadmitted callback")

    class Meta(type):
        __eq__ = forbidden
        __hash__ = forbidden

    class ForeignString(str, metaclass=Meta):
        __eq__ = forbidden
        __hash__ = forbidden
        __str__ = forbidden
        __repr__ = forbidden

    class ForeignInteger(int, metaclass=Meta):
        __eq__ = forbidden
        __hash__ = forbidden
        __int__ = forbidden
        __bool__ = forbidden
        __repr__ = forbidden

    for value in (ForeignString("axon"), ForeignInteger(1)):
        with pytest.raises(ValueError, match="admitted exact primitive"):
            CapturedNativeObject(value).require_native_scalar()


@pytest.mark.parametrize("value", [None, True, 120, "Owner"])
def test_native_constant_type_is_derived_and_survives_serialization(value):
    original = NativeConstantValue(4, (), value)
    assert "declaration" not in {item.name for item in fields(original)}
    restored = pickle.loads(pickle.dumps(original))
    assert restored.require_native_scalar() == value
    assert restored.native_type is type(value)
    changed = replace(original, value=False)
    assert changed.native_type is bool
    with pytest.raises(TypeError):
        replace(original, declaration=NativeTypeDeclaration(str))


@pytest.mark.parametrize("literal", ["None", "True", "123", "'axon'"])
def test_actual_scalar_constant_instruction_retains_its_contents(literal):
    code = compile(f"result = {literal}\n", "authored-scalar.py", "exec")
    load = next(
        item for item in dis.get_instructions(code) if item.opname == "LOAD_CONST"
    )
    stack = NativeOperandStack()
    NativePrimitiveOperation.LOAD_CONST.capture(stack, load)
    value = stack.stack[0]
    assert isinstance(value, NativeConstantValue)
    assert value.instruction_offset == load.offset
    assert type(value.require_native_scalar()) is type(load.argval)
    assert value.require_native_scalar() == load.argval


@pytest.mark.parametrize(
    "source", ["result = 1.25", "result = (1, 2.25)", "def f(): pass"]
)
def test_unsupported_constants_keep_only_type_and_no_executable_code(source):
    code = compile(source, "authored-nonscalar.py", "exec")
    load = next(
        item for item in dis.get_instructions(code) if item.opname == "LOAD_CONST"
    )
    stack = NativeOperandStack()
    NativePrimitiveOperation.LOAD_CONST.capture(stack, load)
    value = stack.stack[0]
    assert type(value) is NativeTypedValue
    assert value.native_type is type(load.argval)
    with pytest.raises(ValueError, match="contents remain unproved"):
        value.require_native_scalar()
    restored = pickle.loads(pickle.dumps(value))
    assert restored.native_type is type(load.argval)
    assert all(not isinstance(item, CodeType) for item in vars(restored).values())


@pytest.mark.parametrize("literal", ["None", "True", "123"])
def test_existing_entry_store_window_does_not_broaden_to_nontext(literal):
    code = compile(f"result = {literal}", "authored-entry.py", "exec")
    window = NativeEntryValueWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    for instruction in dis.get_instructions(code):
        window.observe(instruction)
    assert window.closed
    assert window.load is None
    assert window.store is None


def test_scalar_query_retains_original_producer_and_source_prefix_obligations():
    environment = execution("class Owner:\n    pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    captured = entry.initial_entries["__qualname__"]
    copied_producer = replace(captured.value)
    assert copied_producer is not captured.value
    with pytest.raises(ValueError, match="original"):
        replace(captured, value=copied_producer).require_native_scalar()

    environment = execution("unknown()\nresult = 120\n")
    captured = environment.capture_value(environment.module.module.body[-1].value)
    with pytest.raises(ValueError):
        captured.require_native_scalar()


@pytest.mark.parametrize("literal", ["120", "'known'"])
def test_direct_literal_with_warmed_type_still_requires_original_prefix(literal):
    environment = execution(f"unknown()\nresult = {literal}\n")
    node = environment.module.module.body[-1].value
    read = environment.source.value_reads_by_node[node]
    captured = SourceLiteralCapture(environment, read)
    assert captured.native_type in (int, str)
    for query in (captured.require_native_scalar, captured.require_native_text):
        with pytest.raises(CapturedReferenceRejection) as failure:
            query()
        assert failure.value.violation is CapturedReferenceViolation.UNPROVED_EFFECTS


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 14), reason="3.14 native class metadata"
)
def test_original_firstlineno_contents_match_native_class_and_reject_forgery():
    source = "\n\nclass Owner:\n    pass\n"
    environment = execution(source)
    entry = environment.class_entry(environment.module.module.body[0])
    captured = entry.initial_entries["__firstlineno__"]
    native = subprocess.run(
        [sys.executable, "-c", source + "print(Owner.__firstlineno__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert captured.native_type is int
    assert captured.require_native_scalar() == int(native.stdout)
    with pytest.raises(ValueError, match="original"):
        replace(
            captured, value=replace(captured.value, value=True)
        ).require_native_scalar()
