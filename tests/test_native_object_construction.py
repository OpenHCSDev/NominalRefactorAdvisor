"""Native object allocation shares the existing default-construction protocol."""

from dataclasses import replace
import subprocess
import sys
import weakref

import pytest

from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.native_call import (
    DefaultObjectConstruction,
    NativeObjectConstruction,
)
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceObjectConstruction
from test_source_function_result import execution


@pytest.mark.parametrize(
    "prefix,expression",
    (
        ("", "object()"),
        ("alias = object\n", "alias()"),
        ("from builtins import object as native_object\n", "native_object()"),
    ),
)
@pytest.mark.parametrize("in_class", (False, True))
def test_actual_native_allocation_retains_original_call_and_installation(
    prefix, expression, in_class
):
    source = (
        prefix + ("class Holder:\n    " if in_class else "") + f"held = {expression}\n"
    )
    environment = execution(source)
    owner = environment.module.module.body[-1]
    node = (owner.body[-1] if in_class else owner).value
    context, call = environment.source_call(node)
    result = environment.capture_value(node)
    assert type(result) is NativeObjectConstruction
    assert result.operation is environment.source_operation(context, call)
    assert result.native_type is object
    result.require_closed()
    result.require_release()
    result.require_class_installation()
    environment.required_prefix(context, None)
    if in_class:
        environment.require_class_creation(owner)
    assert not environment._pending
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            source
            + (
                "assert type(Holder.held) is object\n"
                if in_class
                else "assert type(held) is object\n"
            ),
        ],
        check=True,
        timeout=10,
    )


def test_source_and_native_default_construction_share_the_same_execution_owner():
    assert (
        SourceObjectConstruction.require_closed
        is DefaultObjectConstruction.require_closed
    )
    assert (
        NativeObjectConstruction.require_closed
        is DefaultObjectConstruction.require_closed
    )
    assert SourceObjectConstruction.result is NativeObjectConstruction.result
    assert (
        SourceObjectConstruction.native_constructor
        is NativeObjectConstruction.native_declarations[0]
    )
    assert not DefaultObjectConstruction.__abstractmethods__.isdisjoint(
        {"require_instance_hooks"}
    )


@pytest.mark.parametrize("arguments", ("1", "value=1", "*()", "**{}"))
def test_native_argument_protocol_does_not_accept_values_or_unproved_expansion(
    arguments,
):
    environment = execution(f"held = object({arguments})\n")
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[0].value
        ).require_closed()
    assert not environment._pending


@pytest.mark.parametrize(
    "replacement",
    (
        "object = missing\n",
        "def object():\n    raise RuntimeError('analyzer must not run this')\n",
    ),
)
def test_builtin_spelling_does_not_admit_a_rebound_constructor(replacement):
    environment = execution(replacement + "held=object()\n")
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[-1].value
        ).require_closed()
    assert not environment._pending


def test_distinct_allocations_do_not_gain_shared_runtime_identity():
    environment = execution("first=object()\nsecond=object()\nalias=first\n")
    first, second, alias = (
        environment.capture_value(node.value) for node in environment.module.module.body
    )
    assert first is not second
    assert alias is first
    assert not first.proves_same_object(second)
    with pytest.raises(ValueError):
        first.require_native_identity(NativeDeclaration(object))
    other = execution(environment.module.source)
    with pytest.raises(ValueError):
        NativeObjectConstruction(other, first.operation)
    with pytest.raises(ValueError):
        NativeObjectConstruction(environment, replace(first.operation))


@pytest.mark.parametrize(
    "tail",
    (
        "held = None\n",
        "del held\n",
        "values = {}\nvalues['held'] = held\nvalues['held'] = None\n",
    ),
)
def test_exact_object_lifetime_allows_reassignment_and_dictionary_release(tail):
    source = "held=object()\n" + tail + "done = property\n"
    environment = execution(source)
    environment.capture_value(environment.module.module.body[-1].value).require_closed()
    assert not environment._pending
    subprocess.run([sys.executable, "-I", "-S", "-c", source], check=True, timeout=10)


def test_native_lifetime_law_excludes_heap_subclasses_with_owned_state_or_finalizers():
    events = []

    class ActiveObject:
        def __del__(self):
            events.append("released")

    NativeCreationBackend.current().require_inert_instance_release(object)
    CapturedNativeObject(object()).require_release()
    instance = ActiveObject()
    with pytest.raises(ValueError):
        CapturedNativeObject(instance).require_release()
    with pytest.raises(ValueError):
        NativeCreationBackend.current().require_inert_instance_release(ActiveObject)
    assert events == []
    with pytest.raises(TypeError):
        weakref.ref(object())
    del instance
    assert events == ["released"]
