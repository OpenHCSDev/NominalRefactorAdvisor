"""Class construction checks actual member values, including native producers."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source, **bindings):
    module = ParsedModule(
        Path("members.py"), "members", False, ast.parse(source), source
    )
    seed = SourceModuleExecution.from_module(module)
    return SourceModuleExecution(
        SourceModuleEntryPremise(
            seed.source,
            seed.initial,
            dict(
                seed.entry.initial_entries,
                **{
                    name: CapturedNativeObject(value)
                    for name, value in bindings.items()
                },
            ),
            seed.entry.builtins,
        )
    )


@pytest.mark.parametrize(
    "source",
    ("class Owner: pass\n", "class Owner:\n    global __module__\n    pass\n"),
)
def test_compiler_supplied_module_value_hook_is_not_ignored(source):
    calls = []

    class Descriptor:
        def __set_name__(self, owner, name):
            calls.append(name)

    value = Descriptor()
    environment = execution(source, __name__=value)
    with pytest.raises(ValueError):
        environment.require_class_creation(environment.module.module.body[0])
    assert calls == []
    exec(source, {"__name__": value})
    assert calls == ["__module__"]


def test_global_assignment_is_not_a_class_member_installation():
    calls = []

    class Descriptor:
        def __set_name__(self, owner, name):
            calls.append(name)

    value = Descriptor()
    source = "class Owner:\n    global external\n    external = provided\n"
    environment = execution(source, provided=value)
    environment.require_class_creation(environment.module.module.body[0])
    assert not calls
    namespace = {"provided": value}
    exec(source, namespace)
    assert namespace["external"] is value
    assert "external" not in vars(namespace["Owner"])
    assert not calls


def test_nested_class_module_lookup_uses_globals_not_outer_locals():
    source = "class Outer:\n    __name__ = 42\n    class Inner: pass\n"
    environment = execution(source)
    outer = environment.module.module.body[0]
    inner = outer.body[-1]
    environment.require_class_creation(outer)
    entry = environment.class_entry(inner)
    value = entry.initial_entries["__module__"]
    assert value is environment.entry.initial_entries["__name__"]
    assert value.native_type is str


def test_original_native_read_is_required_by_source_admission():
    environment = execution("class Owner: pass\n")
    entry = environment.class_entry(environment.module.module.body[0])
    value = next(
        binding.value
        for binding in entry.capture.prologue.bindings
        if binding.name == "__module__"
    )
    entry.native_value(value).require_closed()
    with pytest.raises(ValueError, match="original"):
        entry.native_value(replace(value))


def test_imported_builtin_class_is_not_a_descriptor_instance():
    source = "class Owner:\n    from builtins import property as descriptor\n"
    environment = execution(source)
    environment.require_class_creation(environment.module.module.body[0])
    namespace = {}
    exec(source, namespace)
    assert namespace["Owner"].descriptor is property
