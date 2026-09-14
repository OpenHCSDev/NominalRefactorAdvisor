"""Actual source consumers retain construction, storage and context obligations."""

import ast
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.native_compilation import NativePrimitiveOperation
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(source: str) -> SourceModuleExecution:
    return SourceModuleExecution.from_module(
        ParsedModule(
            path=Path("source_consumer_probe.py"),
            module_name="source_consumer_probe",
            is_package_init=False,
            module=ast.parse(source),
            source=source,
        )
    )


def _require_final_class(source: str) -> None:
    execution = _execution(source)
    node = execution.module.module.body[-1]
    assert isinstance(node, ast.ClassDef)
    execution.require_class_creation(node)


@pytest.mark.parametrize(
    "source",
    (
        "class Descriptor:\n    def __set_name__(self, owner, name):\n        unknown()\n",
        "class Base: pass\nclass Middle(Base): pass\nclass Leaf(Middle): pass\n",
        "class Base:\n    pass\nclass Middle(Base):\n    pass\nclass Leaf(Middle):\n    pass\n",
        "class Base:\n    pass\nclass Left(Base):\n    pass\nclass Right(Base):\n    pass\nclass Leaf(Left, Right):\n    pass\n",
        "class Annotated:\n    field: int\n",
        "import builtins\nclass Holder:\n    descriptor = builtins.property\n",
        "class Holder:\n    from builtins import property as descriptor\n",
        "saved = 1\nclass saved: pass\n",
    ),
)
def test_native_source_class_completion_preserves_deferred_bodies(source: str) -> None:
    _require_final_class(source)


@pytest.mark.parametrize(
    "source",
    (
        "class Base:\n    def __init_subclass__(cls): unknown()\nclass Leaf(Base): pass\n",
        "class Base:\n    def __init_subclass__(cls): unknown()\nclass Middle(Base): pass\nclass Leaf(Middle): pass\n",
        "class Base: pass\nclass Leaf(Base, metaclass=type): pass\n",
        "def factory(): return object\nclass Leaf(factory()): pass\n",
        "class Base: pass\nBase.__init_subclass__ = print\nclass Leaf(Base): pass\n",
        "class Base: pass\nprint(Base)\nclass Leaf(Base): pass\n",
        "import builtins\ndel builtins.property\nclass Holder: pass\n",
        "import builtins\nbuiltins.property += 1\nclass Holder: pass\n",
        "saved = 1.0\nclass saved: pass\n",
        "del missing\nclass Holder: pass\n",
        "class Base:\n    pass\nclass Invalid(Base, Base):\n    pass\n",
        "class A:\n    pass\nclass B(A):\n    pass\nclass Invalid(A, B):\n    pass\n",
    ),
)
def test_unproved_construction_or_prior_value_release_stays_open(source: str) -> None:
    with pytest.raises(ValueError):
        _require_final_class(source)


def test_replacing_initial_exact_text_metadata_has_native_release_proof() -> None:
    source = "__name__ = 'replacement'\nclass Holder: pass\n"
    _require_final_class(source)
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr


@pytest.mark.parametrize(
    "source",
    (
        "import builtins\nsaved = builtins.property\nsaved = 1\nclass Holder: pass\n",
        "import builtins\nsaved = builtins.property\nimport typing as saved\nclass Holder: pass\n",
    ),
)
def test_exact_static_type_installation_has_safe_later_release(source: str) -> None:
    _require_final_class(source)
    native = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr


def test_equal_context_copy_cannot_reuse_admission_cache() -> None:
    execution = _execution("import builtins\ncaptured = builtins.property\n")
    read = next(read for read in execution.source.reference_reads_by_node.values())
    original = execution.admit(read.context, read.use.position)
    assert not isinstance(original, OpenCapturedReference)
    copied = replace(read.context)
    assert copied is not read.context
    assert isinstance(execution.admit(copied, read.use.position), OpenCapturedReference)


def test_source_owner_lookup_reuses_actual_compact_contexts() -> None:
    execution = _execution("class Owner:\n    pass\n")
    for context in execution.source.compact.flow_contexts:
        assert execution.context_for_owner(context.flow.owner) is context
        with pytest.raises(ValueError, match="unique actual flow context"):
            execution.context_for_owner(replace(context.flow.owner))


def test_foreign_same_named_owner_cannot_select_an_actual_context() -> None:
    source = "class Owner:\n    pass\n"
    execution = _execution(source)
    foreign = _execution(source)
    for context in foreign.source.compact.flow_contexts:
        with pytest.raises(ValueError, match="unique actual flow context"):
            execution.context_for_owner(context.flow.owner)


def test_duplicate_owner_contexts_remain_ambiguous() -> None:
    execution = _execution("class Owner:\n    pass\n")
    source = execution.source
    flow = source.compact.flows[-1]
    execution.entry.__dict__["source"] = replace(
        source, compact=replace(source.compact, flows=(*source.compact.flows, flow))
    )
    with pytest.raises(ValueError, match="unique actual flow context"):
        execution.context_for_owner(flow.owner)


@pytest.mark.parametrize(
    "base, admitted", (("object", True), ("type", False), ("str", False))
)
def test_body_lookup_requires_native_base_protocol_before_entering_body(
    base, admitted
) -> None:
    execution = _execution(
        f"import builtins\nclass Derived(builtins.{base}):\n    captured = builtins.property\n"
    )
    node = execution.module.module.body[-1]
    assert isinstance(node, ast.ClassDef)
    assignment = node.body[0]
    assert isinstance(assignment, ast.Assign)
    captured = execution.capture(assignment.value)
    if admitted:
        captured.require_native_identity(NativeDeclaration(property))
    else:
        assert isinstance(captured, OpenCapturedReference)


def test_native_cell_creation_does_not_fabricate_class_dictionary_bindings() -> None:
    execution = _execution("class Methods:\n    def method(self): return 1\n")
    node = execution.module.module.body[0]
    assert isinstance(node, ast.ClassDef)
    execution.require_class_creation(node)
    entry = execution.class_entry(node)
    for binding in entry.capture.prologue.bindings:
        if binding.operation is NativePrimitiveOperation.MAKE_CELL:
            assert binding.name not in entry.initial_entries
