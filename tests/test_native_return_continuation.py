"""Store continuations retain the actual returned operand and later native writes."""

import ast
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeBindingTransfer,
    NativeContinuationWindow,
    NativeCreationBackend,
    NativeOperandStack,
    NativePrimitiveOperation,
    NativePythonCompilation,
    NativeReturn,
    NativeValueStoreStream,
)
from test_native_scalar_store import store_for
from test_registry_native_registration_controls import native_control


def compiled_stores(source):
    compilation = NativePythonCompilation(source, "return.py")
    nodes = sorted(
        (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)),
        key=lambda node: node.lineno,
    )
    return compilation, tuple(store_for(compilation, node) for node in nodes)


@pytest.mark.parametrize("in_class", (False, True))
def test_return_is_actual_native_none_and_shared_by_original_stores(in_class):
    body = "one = 1\ntwo = None\nthree = True\n"
    source = (
        "class Family:\n" + "".join("    " + line for line in body.splitlines(True))
        if in_class
        else body
    )
    compilation, stores = compiled_stores(source)
    continuation = stores[0].require_return()
    assert all(store.require_return() is continuation for store in stores)
    assert (
        continuation.value.require_scalar_store_value().require_native_scalar() is None
    )
    assert continuation.frame is stores[0].frame
    assert all(continuation.continues(store.binding) for store in stores)
    code = compilation.compile()
    if in_class:
        code = next(value for value in code.co_consts if isinstance(value, CodeType))
    instructions = tuple(dis.get_instructions(code))
    assert continuation.instruction_offset == instructions[-1].offset
    later = continuation.after(stores[0])
    assert {"two", "three"} <= {binding.name for binding in later}
    assert all(
        binding.instruction_offset > stores[0].binding.instruction_offset
        for binding in later
    )


def test_compiler_overwrite_is_not_lost_after_final_source_store():
    compilation, (store,) = compiled_stores(
        "class Family:\n    __static_attributes__ = None\n"
    )
    continuation = store.require_return()
    code = next(
        value
        for value in compilation.compile().co_consts
        if isinstance(value, CodeType)
    )
    actual = tuple(
        instruction.offset
        for instruction in dis.get_instructions(code)
        if instruction.opname == "STORE_NAME"
        and instruction.argval == "__static_attributes__"
        and instruction.offset > store.binding.instruction_offset
    )
    projected = tuple(
        binding.instruction_offset
        for binding in continuation.after(store)
        if binding.name == "__static_attributes__"
    )
    assert projected == actual
    assert store.value.require_native_scalar() is None


def test_return_and_compiler_transfer_match_an_authored_native_execution():
    source = "class Family:\n    __static_attributes__ = None\n"
    _, (store,) = compiled_stores(source)
    continuation = store.require_return()
    outcome = native_control(
        "import json, sys\n"
        "observations = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Family' and event == 'return':\n"
        "        observations.append({'returned': value, 'types': "
        "{name: type(value).__name__ for name, value in frame.f_locals.items()}})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({source!r})\n"
        "sys.settrace(None)\n"
        "assert len(observations) == 1\n"
        "print(json.dumps(observations[0]))\n",
        False,
    )
    assert outcome["returned"] is continuation.value.require_native_scalar()
    for binding in continuation.after(store):
        assert outcome["types"][binding.name] == binding.value.native_type.__name__


@pytest.mark.parametrize(
    "middle", ("unknown + 1", "if condition:\n    unknown()", "raise RuntimeError()")
)
def test_unknown_or_branching_suffix_cannot_borrow_return(middle):
    source = "before = None\n" + middle + "\n"
    _, (before,) = compiled_stores(source)
    with pytest.raises(ValueError):
        before.require_return()


def test_new_straight_line_store_after_unsupported_transfer_has_its_own_continuation():
    _, (before, after) = compiled_stores("before = None\nunknown + 1\nafter = True\n")
    with pytest.raises(ValueError):
        before.require_return()
    assert after.require_return().value.require_native_scalar() is None


def test_foreign_frame_or_replaced_binding_cannot_borrow_continuation():
    _, (store,) = compiled_stores("key = None\n")
    _, (foreign,) = compiled_stores("key = None\n")
    for forged in (
        replace(store, frame=foreign.frame),
        replace(store, binding=replace(store.binding)),
        replace(foreign, continuation=store.continuation),
    ):
        with pytest.raises(ValueError):
            forged.require_return()


def test_native_return_without_result_is_not_accepted():
    instructions = tuple(
        dis.get_instructions(compile("key = None\n", "return.py", "exec"))
    )
    returned = instructions[-1]
    operation = NativeCreationBackend.current().primitive_operations[returned.opcode]
    stack = NativeOperandStack()
    if operation is NativePrimitiveOperation.RETURN_VALUE:
        with pytest.raises(ValueError):
            operation.capture(stack, returned)
    else:
        operation.capture(stack, returned)
        assert stack.returned.require_native_scalar() is None


def test_jump_entry_after_store_invalidates_whole_continuation():
    backend = NativeCreationBackend.current()
    code = compile("key = None\n", "return.py", "exec")
    instructions = list(backend.instructions(code))
    returned = instructions[-1]
    instructions[-1] = (
        returned._replace(label=0)
        if hasattr(returned, "label")
        else returned._replace(is_jump_target=True)
    )
    stream = NativeValueStoreStream(code, backend.primitive_operations)
    for instruction in instructions:
        stream.observe(instruction)
    assert stream.continuation is None


def test_pickling_keeps_one_return_and_original_store_membership(monkeypatch):
    source = "one = None\ntwo = True\n"
    compilation, _ = compiled_stores(source)
    payload = pickle.dumps(compilation)

    def no_compile(self):
        raise AssertionError("Compact return evidence must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    one, two = (store_for(restored, node) for node in ast.parse(source).body)
    assert one.require_return() is two.require_return()
    assert one.require_return().continues(one.binding)
    assert two.require_return().continues(two.binding)


def test_many_stores_share_one_linear_continuation_walk(monkeypatch):
    calls = 0
    original = NativeContinuationWindow.observe

    def observe(self, instruction, operation):
        nonlocal calls
        calls += 1
        return original(self, instruction, operation)

    def no_deep_hash(self):
        raise AssertionError("Store membership must not hash operand graphs")

    monkeypatch.setattr(NativeContinuationWindow, "observe", observe)
    monkeypatch.setattr(NativeBindingTransfer, "__hash__", no_deep_hash)
    source = "".join(f"field_{index} = {index}\n" for index in range(200))
    compilation, stores = compiled_stores(source)
    assert calls <= len(tuple(dis.get_instructions(compilation.compile())))
    continuation = stores[0].require_return()

    def no_transfer_materialisation(self, store):
        raise AssertionError(
            "Membership validation must not materialise later transfers"
        )

    monkeypatch.setattr(NativeReturn, "after", no_transfer_materialisation)
    assert all(store.require_return() is continuation for store in stores)


def test_ambiguous_native_store_membership_is_not_admitted():
    _, (store,) = compiled_stores("key = None\n")
    continuation = store.require_return()
    for duplicate in (store.binding, replace(store.binding)):
        ambiguous = replace(continuation, stores=(store.binding, duplicate))
        assert not ambiguous.continues(store.binding)
        with pytest.raises(ValueError):
            replace(store, continuation=ambiguous).require_return()
