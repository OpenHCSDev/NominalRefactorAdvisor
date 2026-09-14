"""Native class metadata consumes values, not merely their Python types."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import CPythonClassConstructionField
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_native_member_value_origins import execution as bound_execution


@pytest.mark.parametrize(
    "source",
    (
        "class Owner:\n    __doc__ = '\\ud800'\n",
        "class Owner:\n    '\\ud800'\n",
        "text = '\\ud800'\nclass Owner:\n    __doc__ = text\n",
        "class Owner:\n    __doc__ = (text := '\\ud800')\n",
        "class Owner:\n    __doc__ = '\\ud83d\\ude00'\n",
    ),
)
def test_invalid_native_docstring_encoding_cannot_be_admitted(source):
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode != 0
    assert "UnicodeEncodeError" in native.stderr
    with pytest.raises(ValueError):
        environment = SourceModuleExecution.from_module(
            ParsedModule(
                Path("metadata.py"), "metadata", False, ast.parse(source), source
            )
        )
        environment.require_class_creation(environment.module.module.body[-1])


@pytest.mark.parametrize(
    "member",
    (
        "'ordinary docstring'",
        "__doc__ = 'ordinary text'",
        "__doc__ = 'a\\x00b'",
        "__doc__ = '\\U0001f600'",
        "__doc__ = 42",
        "__doc__ = None",
        "__doc__ = [1, 2]",
        "unrelated = '\\ud800'",
    ),
)
def test_valid_class_metadata_and_unrelated_strings_stay_supported(member):
    source = f"class Owner:\n    {member}\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr
    environment = SourceModuleExecution.from_module(
        ParsedModule(Path("metadata.py"), "metadata", False, ast.parse(source), source)
    )
    environment.require_class_creation(environment.module.module.body[0])


@pytest.mark.parametrize("text", ("valid", "\ud800"))
def test_admitted_initial_value_uses_its_actual_text(text):
    source = "class Owner:\n    __doc__ = provided\n"
    environment = bound_execution(source, provided=text)
    if text == "valid":
        environment.require_class_creation(environment.module.module.body[0])
    else:
        with pytest.raises(ValueError):
            environment.require_class_creation(environment.module.module.body[0])


def test_type_only_unicode_evidence_does_not_supply_contents():
    environment = bound_execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    with pytest.raises(ValueError, match="contents remain unproved"):
        CPythonClassConstructionField.DOCUMENTATION.require_value(
            NativeTypePremise(str), entry
        )


def test_unicode_subclass_is_not_mistaken_for_a_nontext_value():
    calls = []

    class Text(str):
        def encode(self, *args, **kwargs):
            calls.append("encode")
            return b"pretend valid"

        def __str__(self):
            calls.append("str")
            return "pretend valid"

    value = Text("\ud800")
    environment = bound_execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    with pytest.raises(ValueError, match="admitted exact primitive"):
        CPythonClassConstructionField.DOCUMENTATION.require_value(
            CapturedNativeObject(value), entry
        )
    assert not calls
    with pytest.raises(UnicodeEncodeError):
        type("Actual", (), {"__doc__": value})
    assert not calls
