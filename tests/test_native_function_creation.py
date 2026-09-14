"""Original raw function creation is stronger than code flags, weaker than entry."""

import ast
from dataclasses import fields, is_dataclass
import dis
import pickle
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    CreatedNativeFunctionExecution,
    ExactNativeFunctionExecution,
    ModuleNativeFrameOrigin,
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativePythonCompilation,
    OpenNativeFrameOrigin,
    OpenNativeFunctionExecution,
    SourceNativeFrameOrigin,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _case(monkeypatch, source, *, damage=None):
    compiled = []
    inventories = []
    native_compile = NativePythonCompilation.compile
    native_inventory = NativeCreationBackend.inventory

    def compile_once(owner):
        code = native_compile(owner)
        compiled.append(code)
        return code

    def inventory_once(backend, code, identity):
        inventory = native_inventory(backend, code, identity)
        if damage is not None:
            damage(inventory)
        inventories.append(inventory)
        return inventory

    monkeypatch.setattr(NativePythonCompilation, "compile", compile_once)
    monkeypatch.setattr(NativeCreationBackend, "inventory", inventory_once)
    compilation = NativePythonCompilation(source, "/repo/function_creation.py")
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "chosen"
    )
    receipt = compilation.execution_for(SourceByteSpan.require_node(node))
    assert len(compiled) == len(inventories) == 1
    return compilation, receipt, compiled[0], inventories[0]


@pytest.mark.parametrize(
    "header,body",
    (
        ("def chosen():", "return None"),
        ("def chosen():", "yield 1"),
        ("async def chosen():", "return None"),
        ("async def chosen():", "yield 1"),
        ("def chosen(value: int = 3) -> int:", "return value"),
    ),
)
def test_creation_uses_same_actual_code_and_original_instruction(
    monkeypatch, header, body
):
    source = f"{header}\n    {body}\n"
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, CreatedNativeFunctionExecution)
    site = receipt.require_creation()
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert emission.containing_code is code
    assert emission.creation.opcode == dis.opmap["MAKE_FUNCTION"]
    assert emission.creation.offset == site.instruction_offset
    assert site.frame is emission.frame_origin
    assert isinstance(site.frame, ModuleNativeFrameOrigin)
    assert site.frame.compilation is compilation.identity
    namespace = {}
    exec(code, namespace)  # Only this controlled fixture; use the SAME compiled code.
    assert emission.code is namespace["chosen"].__code__
    assert receipt.native_flags == namespace["chosen"].__code__.co_flags
    assert compilation.execution_for(receipt.source_span) is receipt


def test_nested_creation_keeps_exact_parent_receipt_without_claiming_activation(
    monkeypatch,
):
    source = "def outer():\n    def chosen(): pass\n    return chosen\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    site = receipt.require_creation()
    assert isinstance(site.frame, SourceNativeFrameOrigin)
    (parent,) = tuple(
        item for item in inventory.emissions if item.receipt is site.frame.execution
    )
    (child,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert child.containing_code is parent.code
    namespace = {}
    exec(code, namespace)
    assert "chosen" not in namespace
    assert namespace["outer"]().__code__ is child.code


def test_decorated_result_can_replace_original_created_function(monkeypatch):
    source = "@replace\ndef chosen(): pass\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    raw = []
    replacement = object()

    def replace(function):
        raw.append(function)
        return replacement

    namespace = {"replace": replace}
    assert raw == []
    receipt.require_creation()
    exec(code, namespace)
    assert namespace["chosen"] is replacement
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert emission.code is raw[0].__code__


@pytest.mark.parametrize("damage", ("missing", "range"))
def test_exact_flags_do_not_fill_missing_creation_evidence(monkeypatch, damage):
    def remove_evidence(inventory):
        for emission in inventory.emissions:
            if damage == "missing":
                emission.creation = None
            else:
                emission.creation = emission.creation._replace(
                    positions=dis.Positions(None, None, None, None)
                )

    _, receipt, code, _ = _case(
        monkeypatch, "def chosen(): pass\n", damage=remove_evidence
    )
    assert type(receipt) is ExactNativeFunctionExecution
    namespace = {}
    exec(code, namespace)
    assert receipt.native_flags == namespace["chosen"].__code__.co_flags
    with pytest.raises(ValueError, match="creation"):
        receipt.require_creation()


def test_unadmitted_backend_keeps_code_flags_without_creation_proof(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    _, receipt, _, _ = _case(monkeypatch, "def chosen(): pass\n")
    assert type(receipt) is ExactNativeFunctionExecution
    with pytest.raises(ValueError, match="creation"):
        receipt.require_creation()


@pytest.mark.parametrize(
    "source,reason",
    (
        (
            "if False:\n    def chosen(): pass\n",
            NativeExecutionUnavailable.NO_EMITTED_CODE,
        ),
        (
            "try:\n    work()\nfinally:\n    def chosen(): pass\n",
            NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN,
        ),
    ),
)
def test_eliminated_or_duplicated_creation_remains_open(monkeypatch, source, reason):
    _, receipt, _, _ = _case(monkeypatch, source)
    assert isinstance(receipt, OpenNativeFunctionExecution)
    assert receipt.violation is reason
    with pytest.raises(ValueError, match="creation"):
        receipt.require_creation()


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native generic functions require Python 3.12+"
)
def test_generic_body_creation_does_not_guess_source_creator_activation(monkeypatch):
    _, receipt, _, _ = _case(
        monkeypatch, "def chosen[T](value: T) -> T:\n    return value\n"
    )
    assert isinstance(receipt, CreatedNativeFunctionExecution)
    assert isinstance(receipt.require_creation().frame, OpenNativeFrameOrigin)


def test_compact_creation_pickle_retains_sharing_without_executable_objects(
    monkeypatch,
):
    compilation, _, _, _ = _case(
        monkeypatch, "def outer():\n    def chosen(): pass\n    return chosen\n"
    )
    restored = pickle.loads(pickle.dumps(compilation))
    pending = [restored]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        assert not isinstance(value, (CodeType, ast.AST, dis.Instruction))
        if is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
            pending.extend(vars(value).values())
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    assert restored._execution_outcome == compilation._execution_outcome
