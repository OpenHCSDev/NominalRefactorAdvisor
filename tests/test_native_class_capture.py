"""Native class captures, independently of source-prefix/frame admission."""

import ast
import builtins
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass, replace
import dis
import json
import multiprocessing
import pickle
import subprocess
import sys
from types import CodeType, FunctionType

import pytest

from nominal_refactor_advisor.native_compilation import (
    ExactNativeClassCapture,
    ModuleNativeFrameOrigin,
    NativeClassCaptureResolverABC,
    NativeCreationBackend,
    NativeCreationInventory,
    NativeExecutionUnavailable,
    NativeFrameOriginResolverABC,
    NativePythonCompilation,
    OpenNativeClassCapture,
    OpenNativeFrameOrigin,
    SourceNativeFrameOrigin,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class BodyInvocation:
    name: str
    body: FunctionType
    caller_code: CodeType


def _compile_case(monkeypatch, source):
    native_compile = NativePythonCompilation.compile
    native_inventory = NativeCreationBackend.inventory
    codes = []
    inventories = []

    def capture_compile(compilation):
        code = native_compile(compilation)
        codes.append(code)
        return code

    def capture_inventory(backend, code, identity):
        result = native_inventory(backend, code, identity)
        inventories.append(result)
        return result

    monkeypatch.setattr(NativePythonCompilation, "compile", capture_compile)
    monkeypatch.setattr(NativeCreationBackend, "inventory", capture_inventory)
    compilation = NativePythonCompilation(source, "native_class_capture_case.py")
    spans = tuple(
        SourceByteSpan.require_node(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
    )
    receipts = tuple(compilation.class_capture_for(span) for span in spans)
    (code,) = codes
    (inventory,) = inventories
    return compilation, code, inventory, receipts


def _execute(code, **extra):
    invocations = []

    def builder(body, name, *bases, **keywords):
        invocations.append(BodyInvocation(name, body, sys._getframe(1).f_code))
        return builtins.__build_class__(body, name, *bases, **keywords)

    namespace = {
        "__name__": "native_class_capture_case",
        "__builtins__": dict(vars(builtins), __build_class__=builder),
        **extra,
    }
    # Execute only this trusted test fixture, with the SAME compiled code used
    # by the production receipt. Never modify pytest's actual builtin dictionary.
    exec(code, namespace)
    return namespace, invocations


def _assert_native_capture(receipt, inventory, invocation):
    assert isinstance(receipt, ExactNativeClassCapture)
    (emission,) = tuple(
        item for item in inventory.emissions if item.receipt is receipt.body
    )
    assert emission.code is invocation.body.__code__
    assert emission.containing_code is invocation.caller_code
    assert emission.creation.offset == receipt.creation.instruction_offset
    (builder,) = tuple(
        item
        for item in inventory.builder_loads
        if item.source_span == receipt.source_span
    )
    assert builder.containing_code is invocation.caller_code
    assert builder.instruction.offset == receipt.builder.instruction_offset
    assert builder.instruction.opcode == dis.opmap["LOAD_BUILD_CLASS"]
    assert receipt.builder.instruction_offset < receipt.creation.instruction_offset
    assert receipt.builder.frame is receipt.creation.frame
    assert receipt.creation is receipt.body.require_creation()


@pytest.mark.parametrize(
    "source",
    (
        "class Target: pass\n",
        "@decorate\nclass Target: pass\n",
        "def create():\n    class Target: pass\n    return Target\nTarget = create()\n",
        "def create(value):\n    class Target:\n        saved = value\n    return Target\nTarget = create(7)\n",
    ),
    ids=("module", "decorated", "function-local", "closure"),
)
def test_capture_joins_actual_native_function_and_loader(monkeypatch, source):
    compilation, code, inventory, (receipt,) = _compile_case(monkeypatch, source)
    namespace, (invocation,) = _execute(code, decorate=lambda value: value)
    _assert_native_capture(receipt, inventory, invocation)
    assert compilation.class_capture_for(receipt.source_span) is receipt
    if invocation.caller_code is code:
        assert isinstance(receipt.builder.frame, ModuleNativeFrameOrigin)
    else:
        assert isinstance(receipt.builder.frame, SourceNativeFrameOrigin)
        (owner,) = tuple(
            item
            for item in inventory.emissions
            if item.receipt is receipt.builder.frame.execution
        )
        assert owner.code is invocation.caller_code
    assert namespace["Target"].__name__ == "Target"


def test_nested_and_repeated_source_owners_are_not_joined_by_name(monkeypatch):
    source = "class Same: pass\n" "First = Same\n" "class Same:\n    class Same: pass\n"
    _, code, inventory, receipts = _compile_case(monkeypatch, source)
    namespace, invocations = _execute(code)
    for receipt in receipts:
        (invocation,) = tuple(
            invocation
            for invocation in invocations
            if any(
                emission.code is invocation.body.__code__
                and emission.receipt is receipt.body
                for emission in inventory.emissions
            )
        )
        _assert_native_capture(receipt, inventory, invocation)
    assert namespace["First"] is not namespace["Same"]
    nested = next(
        item
        for item in receipts
        if isinstance(item.builder.frame, SourceNativeFrameOrigin)
    )
    outer = next(
        item for item in receipts if item.body is nested.builder.frame.execution
    )
    assert nested.source_span != outer.source_span


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
@pytest.mark.parametrize("qualified", (False, True), ids=("bare", "qualified"))
def test_generic_capture_keeps_body_without_guessing_creator_frame(
    monkeypatch, qualified
):
    decorator = "builtins.property" if qualified else "property"
    source = (
        "import builtins\n"
        "class Target[T]:\n"
        f"    @{decorator}\n"
        "    def value(self): return 7\n"
    )
    _, code, inventory, (receipt,) = _compile_case(monkeypatch, source)
    namespace, (invocation,) = _execute(code)
    _assert_native_capture(receipt, inventory, invocation)
    assert isinstance(receipt.builder.frame, OpenNativeFrameOrigin)
    assert (
        receipt.builder.frame.reason is NativeExecutionUnavailable.UNJOINED_FRAME_ORIGIN
    )
    assert invocation.caller_code is not code
    assert namespace["Target"]().value == 7


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
def test_generic_implicit_base_executes_its_actual_protocol_in_subprocess():
    source = """
import json
import typing

events = []
original = typing._GenericAlias.__mro_entries__
def observe(self, bases):
    events.append("implicit-base-protocol")
    return original(self, bases)
try:
    typing._GenericAlias.__mro_entries__ = observe
    class Target[T]:
        pass
finally:
    typing._GenericAlias.__mro_entries__ = original
print(json.dumps({
    "events": events,
    "actual_base_is_generic": Target.__bases__ == (typing.Generic,),
    "protocol_restored": typing._GenericAlias.__mro_entries__ is original,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "events": ["implicit-base-protocol"],
        "actual_base_is_generic": True,
        "protocol_restored": True,
    }


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
def test_generic_creator_and_body_capture_before_base_rebinding(monkeypatch):
    source = (
        "__builtins__ = before\n"
        "class Target[T](base()):\n"
        "    @property\n    def value(self): return 7\n"
    )
    _, code, inventory, (receipt,) = _compile_case(monkeypatch, source)
    observations = []

    class Base:
        pass

    def before_property(function):
        return ("before", function)

    def after_property(function):
        return ("after", function)

    def builder(body, name, *bases, **keywords):
        caller = sys._getframe(1)
        observations.append(
            (BodyInvocation(name, body, caller.f_code), caller.f_builtins)
        )
        assert body.__globals__["__builtins__"] is after
        return builtins.__build_class__(body, name, *bases, **keywords)

    before = dict(vars(builtins), property=before_property, __build_class__=builder)
    after = dict(vars(builtins), property=after_property)
    namespace = {"__builtins__": dict(vars(builtins)), "before": before}

    def base():
        namespace["__builtins__"] = after
        return Base

    namespace["base"] = base
    exec(code, namespace)
    ((invocation, creator_builtins),) = observations
    _assert_native_capture(receipt, inventory, invocation)
    assert creator_builtins is before
    assert invocation.body.__builtins__ is before
    assert namespace["Target"].value[0] == "before"
    # Creation provenance does not yet prove invocation or the implicit Generic
    # base protocol. In particular this is not a source-parent frame join.
    assert isinstance(receipt.builder.frame, OpenNativeFrameOrigin)
    assert invocation.caller_code is not code


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
@pytest.mark.parametrize(
    "damage", ("missing-builder", "missing-body-creation", "range")
)
def test_generic_damaged_native_capture_cannot_supply_entry(monkeypatch, damage):
    backend = NativeCreationBackend.current()
    native_instructions = type(backend).instructions

    def damaged_instructions(owner, code):
        instructions = tuple(native_instructions(owner, code))
        contains_builder = any(
            event.opcode == dis.opmap["LOAD_BUILD_CLASS"] for event in instructions
        )
        for instruction in instructions:
            if instruction.opcode == dis.opmap["LOAD_BUILD_CLASS"]:
                if damage == "missing-builder":
                    continue
                if damage == "range":
                    instruction = instruction._replace(
                        positions=instruction.positions._replace(col_offset=None)
                    )
            if (
                damage == "missing-body-creation"
                and contains_builder
                and instruction.opcode == dis.opmap["MAKE_FUNCTION"]
            ):
                instruction = instruction._replace(
                    opcode=dis.opmap["NOP"], opname="NOP"
                )
            yield instruction

    monkeypatch.setattr(type(backend), "instructions", damaged_instructions)
    _, _, _, (receipt,) = _compile_case(monkeypatch, "class Target[T]: pass\n")
    assert isinstance(receipt, OpenNativeClassCapture)


@pytest.mark.parametrize(
    "source",
    (
        "class Target: pass\n",
        pytest.param(
            "class Target[T]: pass\n",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12), reason="Native type parameters"
            ),
        ),
    ),
)
def test_missing_creation_ranges_cannot_be_replaced_by_a_synthetic_class_site(
    monkeypatch, source
):
    backend = NativeCreationBackend.current()
    original = type(backend).instructions

    def without_creation_range(owner, code):
        for instruction in original(owner, code):
            if instruction.opcode == dis.opmap["MAKE_FUNCTION"]:
                instruction = instruction._replace(
                    positions=instruction.positions._replace(col_offset=None)
                )
            yield instruction

    monkeypatch.setattr(type(backend), "instructions", without_creation_range)
    _, _, _, (capture,) = _compile_case(monkeypatch, source)
    assert isinstance(capture, OpenNativeClassCapture)


def test_plain_body_captures_rebound_builtins_before_base_expression(monkeypatch):
    source = (
        "import builtins\n"
        "original = __builtins__\n"
        "before = dict(original, property=before_property)\n"
        "after = dict(original, property=after_property)\n"
        "__builtins__ = before\n"
        "def base():\n"
        "    global __builtins__\n"
        "    __builtins__ = after\n"
        "    return object\n"
        "class Target(base()):\n"
        "    @property\n    def value(self): return 7\n"
        "class Later:\n"
        "    @property\n    def value(self): return 9\n"
    )

    def before_property(function):
        return ("before", function)

    def after_property(function):
        return ("after", function)

    _, code, inventory, receipts = _compile_case(monkeypatch, source)
    namespace, invocations = _execute(
        code, before_property=before_property, after_property=after_property
    )
    for receipt, invocation in zip(receipts, invocations, strict=True):
        _assert_native_capture(receipt, inventory, invocation)
    assert invocations[0].body.__builtins__ is namespace["before"]
    assert invocations[1].body.__builtins__ is namespace["after"]
    assert namespace["Target"].value[0] == "before"
    assert namespace["Later"].value[0] == "after"
    assert all(
        isinstance(item.builder.frame, ModuleNativeFrameOrigin) for item in receipts
    )


@pytest.mark.parametrize(
    "source, reason",
    (
        (
            "if False:\n    class Target: pass\n",
            NativeExecutionUnavailable.NO_EMITTED_CODE,
        ),
        (
            "def create(flag):\n    try:\n        if flag: return 1\n    finally:\n        class Target: pass\n",
            NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN,
        ),
    ),
    ids=("eliminated", "duplicate-finally"),
)
def test_missing_or_multiple_native_sites_remain_open(monkeypatch, source, reason):
    _, _, _, (receipt,) = _compile_case(monkeypatch, source)
    assert isinstance(receipt, OpenNativeClassCapture)
    assert receipt.reason is reason


def test_duplicate_actual_emission_is_not_collapsed_by_code_identity(monkeypatch):
    compilation, code, inventory, (receipt,) = _compile_case(
        monkeypatch, "class Target: pass\n"
    )
    assert isinstance(receipt, ExactNativeClassCapture)
    (body,) = tuple(
        item for item in inventory.emissions if item.receipt is receipt.body
    )
    duplicated = NativeCreationInventory(
        code, (*inventory.emissions, body), inventory.builder_loads, inventory.prefixes
    )
    result = duplicated.class_captures(
        compilation.identity, (), NativeCreationBackend.current()
    )
    assert isinstance(result[receipt.source_span], OpenNativeClassCapture)
    assert (
        result[receipt.source_span].reason
        is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
    )


def test_repeated_source_owner_code_remains_distinct_from_unjoined_origin(monkeypatch):
    compilation, _, inventory, (outer, inner) = _compile_case(
        monkeypatch, "class Outer:\n    class Inner: pass\n"
    )
    (parent,) = tuple(
        item for item in inventory.emissions if item.receipt is outer.body
    )
    duplicated_owner = replace(parent)
    results = inventory.class_captures(
        compilation.identity,
        (*inventory.emissions, duplicated_owner),
        NativeCreationBackend.current(),
    )
    frame = results[inner.source_span].builder.frame
    assert isinstance(frame, OpenNativeFrameOrigin)
    assert frame.reason is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN


def test_rebinding_replaces_receipt_lifetime_without_rewriting_old_proof(monkeypatch):
    compilation, _, inventory, (outer, inner) = _compile_case(
        monkeypatch, 'class Outer:\n    class Inner:\n        "documentation"\n'
    )
    parent = next(item for item in inventory.emissions if item.receipt is outer.body)
    child = next(item for item in inventory.emissions if item.receipt is inner.body)
    original_binding = child.binding
    original = inner.body
    assert isinstance(original.require_creation().frame, SourceNativeFrameOrigin)
    results = inventory.class_captures(
        compilation.identity,
        (*inventory.emissions, replace(parent)),
        NativeCreationBackend.current(),
    )
    changed = results[inner.source_span]
    assert child.binding is not original_binding
    assert child.receipt is child.binding.receipt is changed.body
    assert changed.body is not original
    assert changed.body.require_creation().frame is changed.builder.frame
    assert isinstance(changed.builder.frame, OpenNativeFrameOrigin)
    (store,) = inventory.constant_stores
    assert store.frame.execution is changed.body
    assert original_binding.receipt is original
    assert isinstance(original.require_creation().frame, SourceNativeFrameOrigin)

    restored = inventory.class_captures(
        compilation.identity, inventory.emissions, NativeCreationBackend.current()
    )[inner.source_span]
    assert child.receipt is restored.body
    assert isinstance(restored.builder.frame, SourceNativeFrameOrigin)
    assert restored.body.require_creation().frame is restored.builder.frame
    assert isinstance(changed.body.require_creation().frame, OpenNativeFrameOrigin)


def test_unsupported_backend_and_rejected_compile_stay_explicit(monkeypatch):
    compilation = NativePythonCompilation("class Target: pass\n", "unsupported.py")
    span = SourceByteSpan.require_node(ast.parse(compilation.source).body[0])
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    receipt = compilation.class_capture_for(span)
    assert isinstance(receipt, OpenNativeClassCapture)
    assert receipt.reason is NativeExecutionUnavailable.UNSUPPORTED_COMPILER
    invalid = NativePythonCompilation("return\n", "rejected.py").class_capture_for(span)
    assert isinstance(invalid, OpenNativeClassCapture)
    assert invalid.reason is NativeExecutionUnavailable.COMPILATION_REJECTED


def test_incomplete_builder_geometry_does_not_admit_an_exact_capture(monkeypatch):
    backend = NativeCreationBackend.current()
    native_instructions = type(backend).instructions

    def incomplete_instructions(owner, code):
        for instruction in native_instructions(owner, code):
            if instruction.opcode == dis.opmap["LOAD_BUILD_CLASS"]:
                instruction = instruction._replace(
                    positions=instruction.positions._replace(col_offset=None)
                )
            yield instruction

    monkeypatch.setattr(type(backend), "instructions", incomplete_instructions)
    _, _, _, (receipt,) = _compile_case(monkeypatch, "class Target: pass\n")
    assert isinstance(receipt, OpenNativeClassCapture)
    assert receipt.reason is NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES


def test_compact_cached_results_pickle_without_native_code(monkeypatch):
    compilation, _, _, receipts = _compile_case(
        monkeypatch, "class Outer:\n    class Inner: pass\n"
    )
    restored = pickle.loads(pickle.dumps(compilation))
    assert (
        tuple(restored.class_capture_for(item.source_span) for item in receipts)
        == receipts
    )
    for item in receipts:
        result = restored.class_capture_for(item.source_span)
        assert result.builder.frame is result.creation.frame
    pending = [compilation]
    seen = set()
    while pending:
        item = pending.pop()
        assert not isinstance(item, (ast.AST, CodeType, FunctionType, dis.Instruction))
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, member.name) for member in fields(item))
            pending.extend(vars(item).values())
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)


def _query_capture_in_child(compilation, span):
    return compilation.class_capture_for(span)


def test_queried_compilation_crosses_spawn_boundary(monkeypatch):
    compilation, _, _, (receipt,) = _compile_case(monkeypatch, "class Target: pass\n")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        restored = executor.submit(
            _query_capture_in_child, compilation, receipt.source_span
        ).result(timeout=20)
    assert restored == receipt
    assert restored.builder.frame is restored.creation.frame


class KeepOrigin(NativeFrameOriginResolverABC[object]):
    def _module_frame_origin_resolution(self, origin):
        return origin

    def _source_frame_origin_resolution(self, origin):
        return origin

    def _open_frame_origin_resolution(self, origin):
        return origin


class KeepCapture(NativeClassCaptureResolverABC[object]):
    def _exact_class_capture_resolution(self, capture):
        return capture

    def _open_class_capture_resolution(self, capture):
        return capture


def test_receipts_own_dispatch_and_preserve_actual_proof_objects(monkeypatch):
    _, _, _, (receipt,) = _compile_case(monkeypatch, "class Target: pass\n")
    assert receipt.resolve(KeepCapture()) is receipt
    assert receipt.builder.frame.resolve(KeepOrigin()) is receipt.builder.frame
