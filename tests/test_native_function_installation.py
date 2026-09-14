"""Original raw-function installation is not source activation or final state."""

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
    InstalledNativeFunctionExecution,
    InstallGlobalNativeFunction,
    InstallNativeFunction,
    ModuleNativeFrameOrigin,
    NativeCreationBackend,
    NativeCreationOperation,
    NativePrimitiveOperation,
    SourceNativeFrameOrigin,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_function_creation import _case


@pytest.mark.parametrize(
    "header,body",
    (
        ("def chosen():", "return None"),
        ("def chosen(value=3, *, named=4):", "return value"),
        ("def chosen(value: int = 3) -> int:", "return value"),
        ("async def chosen():", "return None"),
        ("def chosen():", "yield 1"),
        ("async def chosen():", "yield 1"),
    ),
)
def test_installation_joins_actual_raw_body_and_immediate_original_name_store(
    monkeypatch, header, body
):
    source = f"{header}\n    {body}\n"
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    installation = receipt.require_installation()
    assert installation is receipt.installation
    assert installation.operation is NativePrimitiveOperation.STORE_NAME
    assert installation.name == "chosen"
    assert isinstance(receipt.require_creation().frame, ModuleNativeFrameOrigin)
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert emission.installation is installation
    instructions = inventory.prefixes[id(code)].instructions
    (store,) = tuple(
        item for item in instructions if item.offset == installation.instruction_offset
    )
    assert store.opname == "STORE_NAME" and store.argval == "chosen"
    producer = emission.attachments[-1] if emission.attachments else emission.creation
    assert producer is not None
    producer_index = next(
        index for index, item in enumerate(instructions) if item is producer
    )
    assert instructions[producer_index + 1] is store
    assert compilation.execution_for(receipt.source_span) is receipt
    namespace = {}
    exec(code, namespace)  # Only this authored fixture, never invoke its body.
    assert namespace["chosen"].__code__ is emission.code
    assert receipt.source_span == SourceByteSpan.require_node(ast.parse(source).body[0])


def test_method_installation_retains_original_class_body_frame(monkeypatch):
    source = "class Owner:\n    def chosen(self):\n        return 3\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    site = receipt.require_creation()
    assert isinstance(site.frame, SourceNativeFrameOrigin)
    (method,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    (owner,) = tuple(
        item for item in inventory.emissions if item.receipt is site.frame.execution
    )
    assert method.containing_code is owner.code
    assert receipt.require_installation().name == "chosen"
    namespace = {}
    exec(code, namespace)
    assert vars(namespace["Owner"])["chosen"].__code__ is method.code


def test_decorator_result_store_does_not_install_original_raw_function(monkeypatch):
    source = "@replace\ndef chosen():\n    pass\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, CreatedNativeFunctionExecution)
    receipt.require_creation()
    with pytest.raises(ValueError, match="installation"):
        receipt.require_installation()
    observed = []
    replacement = object()

    def replace(function):
        observed.append(function)
        return replacement

    namespace = {"replace": replace}
    assert observed == []
    exec(code, namespace)
    assert namespace["chosen"] is replacement
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert observed[0].__code__ is emission.code


@pytest.mark.parametrize("take_branch", (False, True))
def test_conditional_native_store_does_not_assert_it_executed(monkeypatch, take_branch):
    source = "if condition:\n    def chosen():\n        pass\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    assert receipt.require_installation().name == "chosen"
    namespace = {"condition": take_branch}
    exec(code, namespace)
    assert ("chosen" in namespace) is take_branch
    if take_branch:
        (emission,) = tuple(
            item for item in inventory.emissions if item.receipt is receipt
        )
        assert namespace["chosen"].__code__ is emission.code


def test_later_reassignment_does_not_change_original_installation_receipt(monkeypatch):
    source = "def chosen():\n    pass\nchosen = 17\n"
    _, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    stores = tuple(
        item
        for item in inventory.prefixes[id(code)].instructions
        if item.opname == "STORE_NAME" and item.argval == "chosen"
    )
    assert len(stores) == 2
    assert receipt.require_installation().instruction_offset == stores[0].offset
    namespace = {}
    exec(code, namespace)
    assert namespace["chosen"] == 17


@pytest.mark.parametrize(
    "source,operation",
    (
        (
            "def outer():\n    def chosen():\n        pass\n    return chosen\n",
            NativePrimitiveOperation.STORE_FAST,
        ),
        (
            "def outer():\n    def chosen():\n        return chosen\n    return chosen\n",
            NativePrimitiveOperation.STORE_DEREF,
        ),
    ),
)
def test_function_local_installation_uses_its_original_native_storage(
    monkeypatch, source, operation
):
    _, receipt, _, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    receipt.require_creation()
    installation = receipt.require_installation()
    assert installation.operation is operation
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert emission.installation is installation
    assert any(
        item.opname == operation.name
        and item.argval == "chosen"
        and item.offset == installation.instruction_offset
        for item in dis.get_instructions(emission.containing_code)
    )


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize("header", ("def chosen():", "async def chosen():"))
def test_global_installation_reuses_the_declared_store_and_original_creation(
    monkeypatch, in_class, header
):
    body = f"global chosen\n{header}\n    return 3\n"
    source = (
        "class Owner:\n" + "".join("    " + line + "\n" for line in body.splitlines())
        if in_class
        else body
    )
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    installation = receipt.require_installation()
    assert installation.operation is NativePrimitiveOperation.STORE_GLOBAL
    assert installation.name == "chosen"
    (emission,) = tuple(item for item in inventory.emissions if item.receipt is receipt)
    assert emission.installation is installation
    assert compilation.execution_for(receipt.source_span) is receipt
    instructions = inventory.prefixes[id(emission.containing_code)].instructions
    index = next(i for i, item in enumerate(instructions) if item is emission.creation)
    actual_store = instructions[index + 1]
    assert actual_store.opname == "STORE_GLOBAL"
    assert actual_store.offset == installation.instruction_offset
    namespace = {}
    exec(code, namespace)
    assert namespace["chosen"].__code__ is emission.code
    if in_class:
        assert "chosen" not in vars(namespace["Owner"])


def test_global_installation_derives_registry_key_and_shares_execution_logic():
    assert (
        InstallGlobalNativeFunction.native_name
        == NativePrimitiveOperation.STORE_GLOBAL.name
    )
    assert InstallNativeFunction.native_name == NativePrimitiveOperation.STORE_NAME.name
    assert (
        InstallGlobalNativeFunction.advance.__func__
        is InstallNativeFunction.advance.__func__
    )
    assert (
        NativeCreationOperation.__registry__["STORE_GLOBAL"]
        is InstallGlobalNativeFunction
    )
    assert NativeCreationOperation.__registry__["STORE_NAME"] is InstallNativeFunction


def test_span_only_backend_does_not_infer_installation_from_code_flags(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    _, receipt, _, _ = _case(monkeypatch, "def chosen(): pass\n")
    assert type(receipt) is ExactNativeFunctionExecution
    with pytest.raises(ValueError, match="installation"):
        receipt.require_installation()


@pytest.mark.parametrize("disturbance", ("intervening", "jump"))
def test_installation_chain_does_not_skip_intervening_or_alternate_entry(
    monkeypatch, disturbance
):
    backend = NativeCreationBackend.current()
    original_instructions = type(backend).instructions

    def disturbed_instructions(self, code):
        instructions = tuple(original_instructions(self, code))
        for instruction in instructions:
            if instruction.opname == "STORE_NAME" and instruction.argval == "chosen":
                if disturbance == "intervening":
                    # Deliberately invalid observation adjacency, not executable
                    # patched bytecode or a source activation assumption.
                    yield instructions[0]._replace(offset=instruction.offset)
                else:
                    instruction = (
                        instruction._replace(label=0)
                        if sys.version_info >= (3, 13)
                        else instruction._replace(is_jump_target=True)
                    )
            yield instruction

    monkeypatch.setattr(type(backend), "instructions", disturbed_instructions)
    _, receipt, _, _ = _case(monkeypatch, "def chosen(): pass\n")
    receipt.require_creation()
    with pytest.raises(ValueError, match="installation"):
        receipt.require_installation()


def test_cached_compact_installation_roundtrip_retains_sharing_without_native_code(
    monkeypatch,
):
    compilation, original, _, _ = _case(
        monkeypatch, "class Owner:\n    def chosen(self): pass\n"
    )
    assert isinstance(original, InstalledNativeFunctionExecution)
    restored = pickle.loads(pickle.dumps(compilation))
    receipt = restored.execution_for(original.source_span)
    assert isinstance(receipt, InstalledNativeFunctionExecution)
    assert restored.execution_for(original.source_span) is receipt
    assert receipt.require_installation() is receipt.installation
    assert receipt.installation == original.installation
    assert receipt.require_creation().frame.execution.compilation is restored.identity
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
