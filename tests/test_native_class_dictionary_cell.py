"""Native construction roles differ from ordinary source dictionary entries."""

import ast
from dataclasses import replace
import subprocess
import sys
from pathlib import Path
from types import CellType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_compilation import CPythonClassConstructionField
from nominal_refactor_advisor.source_execution import (
    SourceFreshCellCapture,
    SourceModuleExecution,
)
from test_native_member_value_origins import execution as bound_execution


@pytest.mark.parametrize("value", ("1", "None", "object", "'ordinary text'"))
def test_class_dictionary_cell_follows_actual_runtime_construction(value):
    source = f"class Owner:\n    __classdictcell__ = {value}\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    environment = SourceModuleExecution.from_module(
        ParsedModule(Path("cell.py"), "cell", False, ast.parse(source), source)
    )
    owner = environment.module.module.body[0]
    if native.returncode:
        assert "TypeError: __classdictcell__ must be a nonlocal cell" in native.stderr
        with pytest.raises(ValueError, match="class construction.*__classdictcell__"):
            environment.require_class_creation(owner)
    else:
        environment.require_class_creation(owner)


def test_global_storage_does_not_acquire_a_class_dictionary_cell_role():
    source = "class Owner:\n    global __classdictcell__\n    __classdictcell__ = 1\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr
    environment = SourceModuleExecution.from_module(
        ParsedModule(Path("cell.py"), "cell", False, ast.parse(source), source)
    )
    environment.require_class_creation(environment.module.module.body[0])


@pytest.mark.parametrize("annotation", ("int", "list"))
def test_original_compiler_cell_does_not_block_class_creation(annotation):
    source = f"class Owner:\n    field: {annotation}\n"
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr
    environment = SourceModuleExecution.from_module(
        ParsedModule(Path("cell.py"), "cell", False, ast.parse(source), source)
    )
    environment.require_class_creation(environment.module.module.body[0])


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred class annotations"
)
def test_class_cell_requires_original_production_and_destination():
    source = "class Owner:\n    field: int\nclass Other: pass\n"
    environment = SourceModuleExecution.from_module(
        ParsedModule(Path("cell.py"), "cell", False, ast.parse(source), source)
    )
    owner, other = map(environment.class_entry, environment.module.module.body)
    field = CPythonClassConstructionField.CLASS_DICTIONARY_CELL
    value = owner.completion_member(field.value)
    assert isinstance(value, SourceFreshCellCapture)
    value.require_class_construction_field(field, owner)
    with pytest.raises(ValueError, match="foreign"):
        value.require_class_construction_field(field, other)
    with pytest.raises(ValueError, match="original"):
        replace(value, value=replace(value.value)).require_class_construction_field(
            field, owner
        )
    with pytest.raises(ValueError, match="class construction"):
        value.require_class_construction_field(
            CPythonClassConstructionField.CLASS_CELL, owner
        )


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred class annotations"
)
def test_supplied_cell_does_not_acquire_compiler_creation_evidence():
    source = "class Other:\n    __classdictcell__ = provided\n"
    cell = CellType(None)
    environment = bound_execution(source, provided=cell)
    with pytest.raises(ValueError, match="class construction"):
        environment.require_class_creation(environment.module.module.body[-1])
    assert cell.cell_contents is None
    namespace = {"provided": cell}
    exec(source, namespace)
    assert cell.cell_contents == dict(vars(namespace["Other"]))
