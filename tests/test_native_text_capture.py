"""Text contents retain their original producer without claiming object identity."""

import ast
from dataclasses import replace
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.native_compilation import NativeConstantValue
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(Path("text.py"), "text", False, ast.parse(source), source)
    )


def test_prologue_text_requires_its_original_producer():
    environment = execution("class Owner:\n    'original docstring'\n")
    entry = environment.class_entry(environment.module.module.body[0])
    captured = entry.initial_entries["__qualname__"]
    assert captured.require_native_text() == "Owner"
    assert not captured.proves_same_object(CapturedNativeObject("Owner"))
    with pytest.raises(ValueError, match="original"):
        replace(
            captured, value=replace(captured.value, value="different")
        ).require_native_text()


def test_text_content_survives_compact_serialization_with_original_links():
    environment = execution("class Owner:\n    'original docstring'\n")
    entry = environment.class_entry(environment.module.module.body[0])
    prologue = pickle.loads(pickle.dumps(entry.capture.prologue))
    value = next(
        binding.value for binding in prologue.bindings if binding.name == "__qualname__"
    )
    assert isinstance(value, NativeConstantValue)
    prologue.require_value(value)
    assert value.require_native_text() == "Owner"
    assert value.native_type is str


@pytest.mark.parametrize(
    "source", ("result = 'original'\n", "text = 'original'\nresult = text\n")
)
def test_literal_and_alias_share_original_content_without_identity_claim(source):
    environment = execution(source)
    value = environment.capture_value(environment.module.module.body[-1].value)
    assert value.require_native_text() == "original"
    assert not value.proves_same_object(CapturedNativeObject("original"))


def test_nontext_capture_does_not_acquire_text_via_conversion():
    environment = execution("result = 42\n")
    value = environment.capture_value(environment.module.module.body[-1].value)
    with pytest.raises(ValueError, match="exact Unicode"):
        value.require_native_text()
