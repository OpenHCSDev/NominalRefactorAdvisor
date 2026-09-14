"""Native module writes close old-value release without analyzer liveness."""

import ast
import builtins
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceViolation,
    InitialNativeFrame,
    InitialNativeIsland,
    NativeNamespace,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.native_compilation import (
    CPythonStaticTypeRequirement,
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import (
    CompactAttributeTarget,
    CompactFlowContext,
    source_product_flow_projection,
)


@dataclass(frozen=True)
class WriteFixtureEffects(CapturedReferenceEffectsABC):
    """Read-only source fixture; direct writes remain the kernel's obligation.

    Prefixes contain only exact-string names, plain module attributes and native
    type RHS bindings. No source code is executed or arbitrary object metadata
    read. Tests requesting a second write must still prove intervening slots.
    """

    source: ParsedModule
    context: CompactFlowContext
    frame: InitialNativeFrame

    def admit(self, context, position):
        allowed = {ast.Module, ast.Assign, ast.Name, ast.Attribute, ast.Load, ast.Store}
        if context is not self.context or any(
            type(node) not in allowed for node in ast.walk(self.source.module)
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        return SingleFlowPrefix(context, self.frame, position)


def fixture(entries, source="target.member = replacement\n"):
    module = ModuleType("write_fixture")
    vars(module).update(entries)
    globals_storage = {"target": module, "replacement": object}
    initial = InitialNativeIsland((builtins, module), (globals_storage,))
    globals_namespace = initial.namespace_for_storage(globals_storage)
    frame = InitialNativeFrame(
        globals_namespace,
        globals_namespace,
        initial.namespace_for_storage(vars(builtins)),
    )
    parsed = ParsedModule(Path("write.py"), "write", False, ast.parse(source), source)
    projected = source_product_flow_projection(parsed)
    context = projected.compact.flow_contexts[0]
    kernel = CapturedReferenceKernel(
        initial,
        WriteFixtureEffects(parsed, context, frame),
    )
    writes = tuple(
        mutation
        for mutation in context.flow.mutations
        if isinstance(mutation.target, CompactAttributeTarget)
    )
    return module, kernel, context, writes


def require_write(kernel, context, write):
    receiver = kernel._read_use(write.target.receiver_use, context, frozenset())
    receiver.require_attribute_write(
        kernel,
        write.target.attribute_name,
        context,
        write.position,
    )


@pytest.mark.parametrize("entries", ({}, {"member": property}, {"member": type}))
def test_missing_or_static_type_slot_has_closed_release(entries):
    module, kernel, context, (write,) = fixture(entries)
    before = vars(module).copy()
    require_write(kernel, context, write)
    assert vars(module) == before  # Analysis did not perform the requested write.


def test_heap_immutable_type_is_not_static_lifetime():
    flags = type.__getattribute__(re.Pattern, "__flags__")
    assert flags & CPythonStaticTypeRequirement.IMMUTABLETYPE
    assert flags & CPythonStaticTypeRequirement.HEAPTYPE
    _, kernel, context, (write,) = fixture({"member": re.Pattern})
    with pytest.raises(ValueError, match="static type lifetime"):
        require_write(kernel, context, write)


def test_mutable_heap_class_is_not_a_static_type():
    class Payload:
        pass

    with pytest.raises(ValueError, match="static type lifetime"):
        CapturedNativeObject(Payload).require_release()


def test_custom_metaclass_metadata_is_not_queried():
    events = []

    class Meta(type):
        def __getattribute__(cls, name):
            events.append(name)
            return super().__getattribute__(name)

    class Payload(metaclass=Meta):
        pass

    with pytest.raises(ValueError, match="instance lifetime"):
        CapturedNativeObject(Payload).require_release()
    assert events == []


def test_arbitrary_initial_instance_stays_open_despite_analyzer_reference():
    class Payload:
        pass

    retained = Payload()
    _, kernel, context, (write,) = fixture({"member": retained})
    with pytest.raises(ValueError, match="instance lifetime"):
        require_write(kernel, context, write)


def test_actual_active_frame_retains_its_native_builtin_dictionary():
    _, kernel, context, (write,) = fixture({"member": vars(builtins)})
    with pytest.raises(ValueError, match="instance lifetime"):
        CapturedNativeObject(vars(builtins)).require_release()
    require_write(kernel, context, write)


def test_arbitrary_dictionary_is_not_retained_by_unrelated_frame_storage():
    arbitrary = {}
    _, kernel, context, (write,) = fixture({"member": arbitrary})
    with pytest.raises(ValueError, match="instance lifetime"):
        require_write(kernel, context, write)


def test_foreign_namespace_wrapper_cannot_manufacture_active_frame_retention():
    _, kernel, context, (write,) = fixture({"member": vars(builtins)})
    foreign = replace(kernel.effects.frame, builtins=NativeNamespace(vars(builtins)))
    forged = replace(kernel, effects=replace(kernel.effects, frame=foreign))
    with pytest.raises(ValueError, match="different admission"):
        require_write(forged, context, write)


def test_storage_release_never_falls_back_to_same_named_builtin():
    class Payload:
        pass

    _, kernel, context, (write,) = fixture(
        {"property": Payload()}, "target.property = replacement\n"
    )
    with pytest.raises(ValueError, match="instance lifetime"):
        require_write(kernel, context, write)


def test_native_active_frame_dictionary_retention_runs_in_isolated_process():
    source = """
import builtins, gc, weakref
events = []
class Payload:
    def __del__(self): events.append("destroyed")
namespace = vars(builtins).copy()
namespace["payload"] = Payload()
reference = weakref.ref(namespace["payload"], lambda ref: events.append("weakref"))
target = {"__builtins__": namespace, "events": events, "reference": reference}
del namespace
exec("__builtins__ = {}; assert reference() is not None; assert not events", target)
gc.collect()
assert events == ["destroyed", "weakref"]
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("attribute", ("__class__", "__dict__"))
def test_native_module_data_descriptor_write_is_not_dictionary_storage(attribute):
    _, kernel, context, (write,) = fixture(
        {attribute: property},
        f"target.{attribute} = replacement\n",
    )
    with pytest.raises(ValueError, match="capture remains open"):
        require_write(kernel, context, write)


def test_prior_write_does_not_reuse_initial_slot_release_proof():
    _, kernel, context, writes = fixture(
        {"member": property},
        "target.member = replacement\ntarget.member = replacement\n",
    )
    require_write(kernel, context, writes[0])
    with pytest.raises(ValueError, match="destruction remains open"):
        require_write(kernel, context, writes[1])


def test_initial_absence_does_not_hide_prior_slot_write():
    _, kernel, context, writes = fixture(
        {},
        "target.member = replacement\ntarget.member = replacement\n",
    )
    with pytest.raises(ValueError, match="destruction remains open"):
        require_write(kernel, context, writes[1])


def test_foreign_module_and_subclass_receiver_are_unproved_without_hooks():
    events = []

    class Foreign(ModuleType):
        def __setattr__(self, name, value):
            events.append(name)
            return super().__setattr__(name, value)

    _, kernel, context, (write,) = fixture({})
    for value in (ModuleType("unadmitted"), Foreign("subclass")):
        with pytest.raises(ValueError, match="capture remains open"):
            CapturedNativeObject(value).require_attribute_write(
                kernel,
                "member",
                context,
                write.position,
            )
    assert events == []


def test_foreign_string_key_does_not_execute_hash_or_equality():
    events = []

    class Foreign(str):
        def __hash__(self):
            events.append("hash")
            return super().__hash__()

        def __eq__(self, other):
            events.append("eq")
            return super().__eq__(other)

    module, kernel, context, (write,) = fixture({})
    with pytest.raises(TypeError, match="exact string key"):
        CapturedNativeObject(module).require_attribute_write(
            kernel,
            Foreign("member"),
            context,
            write.position,
        )
    assert events == []


def test_unsupported_native_backend_does_not_admit_static_type():
    with pytest.raises(ValueError, match="no admitted runtime proof"):
        SpanOnlyCreationBackend().require_static_type_release(property)


def test_supported_backend_reuses_static_type_lifetime_owner():
    NativeCreationBackend.current().require_static_type_release(property)


@pytest.mark.parametrize("scenario", ("static", "finalizer", "weakref"))
def test_native_release_controls_run_in_isolated_process(scenario):
    programs = {
        "static": """
import gc, weakref
from types import ModuleType
events = []
module = ModuleType("fixture")
module.member = property
reference = weakref.ref(module.member, lambda value: events.append("weakref"))
module.member = object
gc.collect()
assert reference() is property
assert events == []
""",
        "finalizer": """
from types import ModuleType
events = []
class Payload:
    def __del__(self):
        events.append("finalizer")
module = ModuleType("fixture")
module.member = Payload()
module.member = object
assert events == ["finalizer"]
""",
        "weakref": """
import weakref
from types import ModuleType
events = []
class Payload:
    pass
module = ModuleType("fixture")
module.member = Payload()
reference = weakref.ref(module.member, lambda value: events.append("weakref"))
module.member = object
assert reference() is None
assert events == ["weakref"]
""",
    }
    subprocess.run([sys.executable, "-I", "-c", programs[scenario]], check=True)
