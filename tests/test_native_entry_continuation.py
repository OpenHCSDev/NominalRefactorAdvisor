"""A return suffix is not necessarily a path from the original frame entry."""

import ast
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def body_receipt(body, parameters=""):
    source = (
        f"def chosen({parameters}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    compilation = NativePythonCompilation(source, "entry_continuation.py")
    declaration = compilation.execution_for(
        SourceByteSpan.require_node(ast.parse(source).body[0])
    )
    return compilation, declaration, compilation.return_from(declaration)


@pytest.mark.parametrize(
    "body,parameters",
    (
        ("return None", ""),
        ("pass", ""),
        ("return value", "value"),
        ("result = value\nreturn result", "value"),
        ("result = source[key]\nreturn result", "source, key"),
        ("result = supplied()\nreturn result", "supplied"),
        ("def nested():\n    return None\nreturn nested", ""),
    ),
)
def test_uninterrupted_body_retains_its_original_entry_continuation(body, parameters):
    compilation, declaration, receipt = body_receipt(body, parameters)
    assert receipt.require_from_entry() is receipt
    assert compilation.return_from(declaration) is receipt
    assert receipt.frame.is_body_of(declaration)


@pytest.mark.parametrize(
    "body,parameters",
    (
        ("result = left + right\nreturn None", "left, right"),
        ("if flag:\n    result = 1\nresult = None\nreturn result", "flag"),
        ("for item in items:\n    pass\nresult = None\nreturn result", "items"),
    ),
)
def test_later_native_suffix_does_not_erase_earlier_unproved_control(body, parameters):
    _, _, receipt = body_receipt(body, parameters)
    receipt.require_value(receipt.value)
    with pytest.raises(ValueError, match="entry"):
        receipt.require_from_entry()


def test_entry_continuation_survives_a_snapshot_without_recompiling(monkeypatch):
    compilation, declaration, receipt = body_receipt("return value", "value")
    for value in receipt.values:
        receipt.require_value(value)
    payload = pickle.dumps(compilation)

    def forbidden_compile(self, **kwargs):
        raise AssertionError("A retained native receipt must not be reconstructed")

    monkeypatch.setattr(NativePythonCompilation, "compile", forbidden_compile)
    restored = pickle.loads(payload)
    original = restored.execution_for(declaration.source_span)
    restored_receipt = restored.return_from(original)
    assert restored_receipt is not receipt
    assert restored_receipt.require_from_entry() is restored_receipt
    for value in restored_receipt.values:
        restored_receipt.require_value(value)


@pytest.mark.parametrize("expression", ("(first, second)", "(first, first)"))
def test_paired_local_reads_retain_each_original_value_even_at_one_instruction(
    expression,
):
    _, _, receipt = body_receipt("return " + expression, "first, second")
    receipt.require_from_entry()
    left, right = receipt.value.inputs
    assert left is not right
    for value in (left, right):
        receipt.require_value(value)
        with pytest.raises(ValueError, match="original production"):
            receipt.require_value(replace(value))
    if left.instruction_offset == right.instruction_offset:
        with pytest.raises(ValueError, match="unique original production"):
            receipt.production_at(left.instruction_offset)
