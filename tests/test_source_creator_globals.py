"""Class creation retains real creator globals, independently of entry locals."""

import ast
import builtins
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedEntryContents,
    CapturedFlowPrefix,
    CapturedNativeObject,
    InitialNativeFrame,
    InitialNativeIsland,
    NamespaceCreationEvidenceABC,
    RecordedNamespace,
)
from nominal_refactor_advisor.source_entry import SourceExecutionEntryABC
from nominal_refactor_advisor.source_execution import SourceExecutionABC
from test_source_function_result import execution


@dataclass(eq=False)
class SplitNamespaceEntry(
    NamespaceCreationEvidenceABC, RecordedNamespace, SourceExecutionEntryABC
):
    """Explicit authored exec(source, globals, locals) control premise."""

    source: object
    native_island: InitialNativeIsland
    globals_namespace: object
    builtins_namespace: object
    initial_entries: dict

    @property
    def initial(self):
        return self.native_island

    @property
    def context(self):
        return self.source.module_context

    @cached_property
    def frame(self):
        return InitialNativeFrame(self, self.globals_namespace, self.builtins_namespace)

    def prefix(self, position, kernel):
        return CapturedFlowPrefix(self.context, self.frame, position, kernel=kernel)

    def require_admitted(self, initial):
        if initial is not self.initial:
            raise ValueError("Foreign supplied namespace")

    def _member(self, key):
        return self.initial_entries.get(key)

    def require_external_noninterference(self, prefix):
        raise ValueError("No external interference premise supplied")

    def require_native_behavior(self, authority):
        raise ValueError("No native operation premise supplied")

    def require_operation_completion(self, authority):
        raise ValueError("No completion premise supplied")


class SplitNamespaceExecution(SourceExecutionABC):
    @cached_property
    def initial_contents(self):
        return CapturedEntryContents(self.kernel, self.entry)


@pytest.fixture(params=(False, True))
def separate_namespace_class(request):
    body = "class Chosen:\n    selected = outside\n"
    source = (
        "class Outer:\n" + "\n".join("    " + line for line in body.splitlines()) + "\n"
        if request.param
        else body
    )
    globals_module = ModuleType("real_globals")
    globals_module.__dict__.update(outside=73, __builtins__=vars(builtins))
    local_values = {"outside": 11, "__name__": "local_shadow"}
    native_locals = local_values.copy()
    exec(source, vars(globals_module), native_locals)  # Authored control only.
    native_class = (
        native_locals["Outer"].Chosen if request.param else native_locals["Chosen"]
    )
    assert native_class.selected == 73
    assert native_class.__module__ == "real_globals"

    original = execution(source)
    initial = InitialNativeIsland((builtins, globals_module))
    entry = SplitNamespaceEntry(
        original.source,
        initial,
        initial.namespace_for_storage(vars(globals_module)),
        initial.namespace_for_storage(vars(builtins)),
        {key: CapturedNativeObject(value) for key, value in local_values.items()},
    )
    environment = SplitNamespaceExecution(entry)
    node = next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.ClassDef) and node.name == "Chosen"
    )
    return environment, environment.class_entry(node), native_class


def test_class_frame_globals_are_derived_from_the_actual_creator(
    separate_namespace_class,
):
    environment, selected, _ = separate_namespace_class
    assert selected.frame.globals is environment.entry.frame.globals
    assert selected.frame.globals is not environment.entry
    assert selected.frame.builtins is environment.entry.frame.builtins


def test_class_prologue_looks_up_module_name_in_actual_globals(
    separate_namespace_class,
):
    _, selected, native = separate_namespace_class
    assert (
        selected.initial_entries["__module__"].require_native_text()
        == native.__module__
    )


def test_class_body_value_uses_globals_not_outer_exec_locals(separate_namespace_class):
    environment, selected, native = separate_namespace_class
    captured = environment.capture_value(selected.node.body[0].value)
    assert captured.require_native_scalar() == native.selected
