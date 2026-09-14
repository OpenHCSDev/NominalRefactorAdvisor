"""Class namespace construction uses actual storage and native obligations."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("construction.py"), "construction", False, ast.parse(source), source
        )
    )


@pytest.mark.parametrize("member", ("__slots__", "__classcell__"))
def test_invalid_native_construction_field_cannot_be_admitted(member):
    source = f"class Owner:\n    {member} = 1\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode != 0
    assert "TypeError" in native.stderr
    environment = execution(source)
    with pytest.raises(ValueError, match="class construction"):
        environment.require_class_creation(environment.module.module.body[-1])


@pytest.mark.parametrize("member", ("__slots__", "__classcell__"))
def test_global_storage_is_not_a_class_construction_field(member):
    source = f"class Owner:\n    global {member}\n    {member} = 1\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr
    environment = execution(source)
    environment.require_class_creation(environment.module.module.body[-1])


@pytest.mark.parametrize(
    "source",
    (
        "__slots__ = 1\nclass Owner:\n    pass\n",
        "class Owner:\n    field = 1\n",
        "class Base: pass\nclass Owner(Base):\n    field = object\n",
    ),
)
def test_default_native_namespace_construction_stays_supported(source):
    environment = execution(source)
    environment.require_class_creation(environment.module.module.body[-1])
