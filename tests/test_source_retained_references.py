"""Slot release needs a surviving reference at the actual pre-write cut."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedSlotQuery,
    NativeTypePremise,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    parsed = ParsedModule(
        Path("/repo/retained.py"), "retained", False, ast.parse(source), source
    )
    return SourceModuleExecution.from_module(parsed)


def final_capture(owner):
    return owner.capture(owner.module.module.body[-1].value)


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
@pytest.mark.parametrize(
    "bindings",
    (
        "saved = Original\nOriginal = object",
        "alias = Original\nsaved = alias\nalias = object\nOriginal = object",
        "saved = Original\nalias = Original\nOriginal = alias = object",
    ),
)
def test_surviving_slot_retains_the_actual_definition(declaration, bindings):
    source = f"{declaration}\n{bindings}\nsaved\n"
    native = subprocess.run(
        [sys.executable, "-c", source + "print(saved.__qualname__)\n"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert native.stdout.strip() == "Original"
    owner = execution(source)
    captured = final_capture(owner)
    captured.require_closed()
    _, binding = captured.source_definition()
    assert binding.target.bound_name == "Original"


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
@pytest.mark.parametrize(
    "bindings",
    (
        "Original = object",
        "saved = Original\nsaved = object\nOriginal = object",
        "saved = Original\ndel saved\nOriginal = object",
        "class Other: pass\nsaved = Other\nOriginal = object",
        "Original = Original",
        "saved = Original\nOriginal = saved = object",
    ),
)
def test_destination_and_historical_aliases_are_not_retention_proof(
    declaration, bindings
):
    source = f"{declaration}\n{bindings}\nOriginal\n"
    subprocess.run([sys.executable, "-c", source], check=True)
    with pytest.raises(ValueError):
        final_capture(execution(source)).require_closed()


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
def test_definition_identity_includes_the_actual_execution(declaration):
    first = execution(f"{declaration}\nOriginal\n")
    # Share source observations, but not the supplied activation premise.
    second = SourceModuleExecution.from_source(first.source)
    one = final_capture(first)
    repeat = final_capture(first)
    other = final_capture(second)
    assert one.proves_same_object(repeat)
    assert not one.proves_same_object(other)
    assert one.source_definition()[1] is other.source_definition()[1]
    prefix = first.required_prefix(
        first.entry.context,
        first.source.value_reads_by_node[
            first.module.module.body[-1].value
        ].use.position,
    )
    with pytest.raises(ValueError, match="another execution admission"):
        one.require_release_from(
            second.kernel,
            CapturedSlotQuery(first.entry, "Original", prefix, frozenset()),
        )


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
def test_repeated_spelling_does_not_reuse_creation_identity(declaration):
    source = f"{declaration}\nsaved = Original\n{declaration}\nOriginal\n"
    subprocess.run(
        [sys.executable, "-c", source + "assert saved is not Original\n"], check=True
    )
    owner = execution(source)
    previous = owner.capture(owner.module.module.body[1].value)
    current = final_capture(owner)
    previous.require_closed()
    current.require_closed()
    assert not previous.proves_same_object(current)


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
def test_an_active_class_namespace_can_retain_a_global_slot_value(declaration):
    source = (
        f"{declaration}\nclass Container:\n    global Original\n"
        "    saved = Original\n    Original = object\nContainer\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            source + "assert Container.saved.__name__ == 'Original'\n",
        ],
        check=True,
    )
    final_capture(execution(source)).require_closed()


def test_completed_prepared_class_storage_is_not_a_live_frame_witness():
    source = (
        "class Original: pass\nclass Container:\n    saved = Original\n"
        "Original = object\nContainer\n"
    )
    # The actual class retains it, but proving that needs its live type storage,
    # not the now-finished prepared dictionary used by the class-body frame.
    subprocess.run([sys.executable, "-c", source], check=True)
    with pytest.raises(ValueError):
        final_capture(execution(source)).require_closed()


def test_analyzer_capture_identity_alone_never_proves_runtime_identity():
    premise = NativeTypePremise(type)
    assert not premise.proves_same_object(premise)
