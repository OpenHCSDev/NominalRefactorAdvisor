"""Complete native observations do not extend initial-region proof authority."""

from collections import Counter
import dis
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCodeObservation,
    NativeCreationBackend,
    NativePythonCompilation,
)

CLASS_BODIES = (
    "    pass\n",
    "    value = 1\n",
    "    def method(self):\n        raise AssertionError('never execute')\n",
    "    if condition:\n        left = 1\n    else:\n        right = 2\n",
    "    try:\n        value = source\n    finally:\n        final = 1\n",
    "    raise RuntimeError('never execute')\n",
)


def native_inventory(source):
    compilation = NativePythonCompilation(source, "observation_fixture.py")
    code = compilation.compile()
    backend = NativeCreationBackend.current()
    inventory = backend.inventory(code, compilation.identity)
    body = next(value for value in code.co_consts if isinstance(value, CodeType))
    return backend, inventory, inventory.prefixes[id(body)]


@pytest.mark.parametrize(
    "body",
    CLASS_BODIES,
    ids=("childless-pass", "childless-store", "method", "branch", "finally", "raise"),
)
def test_inventory_records_every_original_class_instruction_in_one_walk(
    monkeypatch, body
):
    backend = NativeCreationBackend.current()
    original_instructions = type(backend).instructions
    observed = {}
    calls = Counter()

    def record_instructions(self, code):
        calls[id(code)] += 1
        instructions = tuple(original_instructions(self, code))
        observed[id(code)] = instructions
        return iter(instructions)

    monkeypatch.setattr(type(backend), "instructions", record_instructions)
    _, inventory, observation = native_inventory("class Target:\n" + body)
    original = observed[id(observation.code)]
    assert len(observation.instructions) == len(original)
    assert all(
        actual is expected
        for actual, expected in zip(observation.instructions, original, strict=True)
    )
    assert calls and all(count == 1 for count in calls.values())
    assert all(
        isinstance(item, NativeCodeObservation) for item in inventory.prefixes.values()
    )
    assert observation.instructions[-1] is original[-1]
    # Recording an instruction is not evidence of successful body execution.
    if "raise RuntimeError" in body:
        assert original[-1].opname == "RAISE_VARARGS"


def test_normal_returns_and_later_finally_handler_are_not_truncated():
    _, _, branch = native_inventory("class Target:\n" + CLASS_BODIES[3])
    expected_returns = tuple(
        item.offset
        for item in dis.get_instructions(branch.code)
        if item.opname in {"RETURN_VALUE", "RETURN_CONST"}
    )
    actual_returns = tuple(
        item.offset
        for item in branch.instructions
        if item.opname in {"RETURN_VALUE", "RETURN_CONST"}
    )
    assert len(expected_returns) >= 2
    assert actual_returns == expected_returns

    _, _, finalizer = native_inventory("class Target:\n" + CLASS_BODIES[4])
    first_return = next(
        item
        for item in finalizer.instructions
        if item.opname in {"RETURN_VALUE", "RETURN_CONST"}
    )
    assert any(
        item.offset > first_return.offset and item.opname == "RERAISE"
        for item in finalizer.instructions
    )
    assert dis.Bytecode(finalizer.code).exception_entries


def test_initial_projection_remains_live_until_body_and_never_resumes():
    _, _, original = native_inventory("class Target:\n    value = 1\n")
    observation = NativeCodeObservation(original.code)
    expected_initial = []
    expected_boundary = None
    header = (original.code.co_firstlineno, original.code.co_firstlineno, 0, 0)
    assert observation.initial_instructions == ()
    assert observation.boundary is None
    for index, instruction in enumerate(original.instructions):
        observation.observe(instruction)
        if expected_boundary is None:
            if None in instruction.positions or tuple(instruction.positions) == header:
                expected_initial.append(instruction)
            else:
                expected_boundary = instruction
        assert len(observation.instructions) == index + 1
        assert observation.instructions[-1] is instruction
        assert observation.boundary is expected_boundary
        assert len(observation.initial_instructions) == len(expected_initial)
        assert all(
            actual is expected
            for actual, expected in zip(
                observation.initial_instructions, expected_initial, strict=True
            )
        )
    assert expected_boundary is not None
    # A later header-shaped event cannot restart the initial region. This is
    # an observation-shape control, not an invented source execution receipt.
    later_header = original.instructions[0]._replace(
        offset=original.instructions[-1].offset + 2
    )
    observation.observe(later_header)
    assert observation.instructions[-1] is later_header
    assert observation.initial_instructions == tuple(expected_initial)
    assert observation.boundary is expected_boundary


def test_full_stream_does_not_authorize_source_method_as_initial_helper():
    _, _, observation = native_inventory("class Target:\n" + CLASS_BODIES[2])
    creation = next(
        item for item in observation.instructions if item.opname == "MAKE_FUNCTION"
    )
    assert all(item is not creation for item in observation.initial_instructions)
    assert all(emission.creation is not creation for emission in observation.emissions)
    with pytest.raises(ValueError, match="outside.*prefix"):
        observation.require_creation(creation)


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 14), reason="3.14 generated class annotation helper"
)
def test_initial_helper_retains_original_instruction_and_emission_identity():
    _, _, observation = native_inventory("class Target:\n    value: int\n")
    (emission,) = observation.emissions
    creation = emission.creation
    assert creation is not None
    assert any(item is creation for item in observation.initial_instructions)
    assert any(item is emission.load for item in observation.initial_instructions)
    assert observation.require_creation(creation) is emission
    copied = creation._replace()
    assert copied == creation and copied is not creation
    with pytest.raises(ValueError, match="outside.*prefix"):
        observation.require_creation(copied)
