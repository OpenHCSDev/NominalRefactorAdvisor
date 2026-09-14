"""Observed definition applications retain raw creation and installed result separately."""

import ast
from copy import copy
from dataclasses import fields, replace
import inspect
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    AppliedNativeFunctionExecution,
    ApplyNativeDefinition,
    AttachNativeFunctionAttribute,
    ContiguousNativeCreationBackend,
    CreatedNativeFunctionOperation,
    InstallGlobalNativeFunction,
    InstallNativeFunction,
    NativeCreationBackend,
    NativeCreationOperation,
    NativeCaptureSite,
    NativeFunctionCreationOperation,
    NativePythonCompilation,
    PrepareNativeDefinitionApplication,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_function_creation import _case


@pytest.mark.parametrize(
    "operation",
    (
        AttachNativeFunctionAttribute,
        PrepareNativeDefinitionApplication,
        ApplyNativeDefinition,
        InstallNativeFunction,
        InstallGlobalNativeFunction,
    ),
)
def test_created_transfer_guard_is_shared_and_rejects_uncreated_input(
    monkeypatch, operation
):
    _, receipt, _, inventory = _case(monkeypatch, "def chosen(): pass\n")
    (original,) = (item for item in inventory.emissions if item.receipt is receipt)
    instruction = original.creation
    uncreated = replace(original, creation=None)
    before = (
        tuple(uncreated.attachments),
        tuple(uncreated.applications),
        tuple(uncreated.application_prelude),
        uncreated.installation,
    )
    assert operation.advance.__func__ is CreatedNativeFunctionOperation.advance.__func__
    assert NativeCreationOperation.__registry__[operation.native_name] is operation
    for current in (None, uncreated):
        assert (
            operation.advance(
                NativeCreationBackend.current(), current, instruction, original
            )
            is None
        )
    assert uncreated.creation is None
    assert (
        tuple(uncreated.attachments),
        tuple(uncreated.applications),
        tuple(uncreated.application_prelude),
        uncreated.installation,
    ) == before


def test_attachment_combines_created_guard_and_function_production_by_mro():
    assert inspect.isabstract(CreatedNativeFunctionOperation)
    assert (
        CreatedNativeFunctionOperation
        not in NativeCreationOperation.__registry__.values()
    )
    assert AttachNativeFunctionAttribute.capture_prologue.__func__ is (
        NativeFunctionCreationOperation.capture_prologue.__func__
    )


@pytest.mark.parametrize("decorators", (("first",), ("first", "second")))
@pytest.mark.parametrize("header", ("def chosen():", "async def chosen():"))
@pytest.mark.parametrize("in_class", (False, True))
def test_application_order_and_result_store_match_actual_execution(
    monkeypatch, decorators, header, in_class
):
    body = "".join(f"@{name}\n" for name in decorators) + header + "\n    return 3\n"
    source = (
        "class Owner:\n" + "".join("    " + line for line in body.splitlines(True))
        if in_class
        else body
    )
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    assert isinstance(receipt, AppliedNativeFunctionExecution)
    (emission,) = (item for item in inventory.emissions if item.receipt is receipt)
    applications = receipt.require_applications()
    assert all(
        isinstance(application, NativeCaptureSite) for application in applications
    )
    assert "site" not in {field.name for field in fields(applications[0])}
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "chosen"
    )
    assert tuple(application.source_span for application in applications) == tuple(
        SourceByteSpan.require_node(decorator)
        for decorator in reversed(node.decorator_list)
    )
    instructions = inventory.prefixes[id(emission.containing_code)].instructions
    calls = tuple(
        instruction
        for instruction in instructions
        if instruction.opname == "CALL" and instruction.arg == 0
    )
    assert tuple(
        application.instruction_offset for application in applications
    ) == tuple(call.offset for call in calls)
    assert all(
        application.frame is receipt.require_creation().frame
        for application in applications
    )
    assert all(
        len(application.prelude)
        == len(NativeCreationBackend.current().application_prelude)
        for application in applications
    )
    assert all(
        site.frame is receipt.require_creation().frame
        for application in applications
        for site in application.prelude
    )

    observed = []
    results = {name: object() for name in decorators}

    def make_decorator(name):
        def decorate(value):
            observed.append((name, value))
            return results[name]

        return decorate

    namespace = {name: make_decorator(name) for name in decorators}
    assert observed == []
    exec(code, namespace)  # Only this authored fixture, never the function body.
    assert tuple(name for name, _ in observed) == tuple(reversed(decorators))
    assert observed[0][1].__code__ is emission.code
    for index in range(1, len(observed)):
        assert observed[index][1] is results[observed[index - 1][0]]
    installed = vars(namespace["Owner"])["chosen"] if in_class else namespace["chosen"]
    assert installed is results[decorators[0]]
    assert installed is not observed[0][1]
    binding = receipt.require_applied_installation()
    assert binding is emission.installation
    continuation = compilation.return_after_binding(binding.source_span, binding.name)
    assert continuation.continues(binding)
    assert not continuation.continues(copy(binding))
    assert continuation.frame is receipt.require_creation().frame
    assert compilation.execution_for(receipt.source_span) is receipt
    with pytest.raises(ValueError, match="raw function installation"):
        receipt.require_installation()
    with pytest.raises(ValueError, match="raw function installation"):
        compilation.return_after(receipt)


def test_global_applied_result_uses_the_original_global_store(monkeypatch):
    source = "class Owner:\n    global chosen\n    @replace\n    def chosen(): pass\n"
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    observed = []
    sentinel = object()

    def decorate(function):
        observed.append(function)
        return sentinel

    namespace = {"replace": decorate}
    exec(code, namespace)
    assert namespace["chosen"] is sentinel
    assert "chosen" not in vars(namespace["Owner"])
    (emission,) = (item for item in inventory.emissions if item.receipt is receipt)
    assert observed[0].__code__ is emission.code
    binding = receipt.require_applied_installation()
    native = next(
        step
        for step in inventory.prefixes[id(emission.containing_code)].instructions
        if step.offset == binding.instruction_offset
    )
    assert native.opname == "STORE_GLOBAL"
    assert compilation.return_after_binding(
        binding.source_span, binding.name
    ).continues(binding)


def test_raw_function_does_not_gain_an_application_receipt(monkeypatch):
    _, receipt, _, _ = _case(monkeypatch, "def chosen(): pass\n")
    receipt.require_installation()
    with pytest.raises(ValueError, match="applied-result"):
        receipt.require_applied_installation()
    with pytest.raises(ValueError, match="applications remain unproved"):
        receipt.require_applications()


@pytest.mark.parametrize(
    "damage",
    (
        "call_arity",
        "missing_preparation",
        "extra_preparation",
        "wrong_preparation_arity",
        "wrong_span",
    ),
)
def test_malformed_native_application_cannot_borrow_result_storage(monkeypatch, damage):
    original = ContiguousNativeCreationBackend.instructions
    backend = NativeCreationBackend.current()
    if "preparation" in damage and not backend.application_prelude:
        pytest.skip("This backend has no preparation instruction")

    def instructions(self, code):
        for step in original(self, code):
            if step.opname == "PRECALL":
                if damage == "missing_preparation":
                    continue
                if damage == "extra_preparation":
                    yield step
                if damage == "wrong_preparation_arity":
                    step = step._replace(arg=1)
            if step.opname == "CALL":
                if damage == "call_arity":
                    step = step._replace(arg=1)
                if damage == "wrong_span":
                    step = step._replace(positions=step.positions._replace(lineno=None))
            yield step

    monkeypatch.setattr(ContiguousNativeCreationBackend, "instructions", instructions)
    _, receipt, _, _ = _case(monkeypatch, "@replace\ndef chosen(): pass\n")
    receipt.require_creation()
    with pytest.raises(ValueError):
        receipt.require_applied_installation()


def test_unsupported_transfer_after_applied_store_does_not_gain_a_return(monkeypatch):
    compilation, receipt, _, _ = _case(
        monkeypatch, "@replace\ndef chosen(): pass\nunknown + 1\n"
    )
    binding = receipt.require_applied_installation()
    with pytest.raises(ValueError):
        compilation.return_after_binding(binding.source_span, binding.name)


def test_application_receipts_survive_serialisation_without_recompiling(monkeypatch):
    source = "@replace\ndef chosen(): pass\n"
    compilation = NativePythonCompilation(source, "applied.py")
    span = SourceByteSpan.require_node(ast.parse(source).body[0])
    receipt = compilation.execution_for(span)
    assert receipt.require_applications()
    payload = pickle.dumps(compilation)

    def no_compile(self):
        raise AssertionError("Original compact application receipts must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    actual = restored.execution_for(span)
    binding = actual.require_applied_installation()
    assert restored.return_after_binding(binding.source_span, binding.name).continues(
        binding
    )
    assert all(
        application.frame is actual.require_creation().frame
        for application in actual.require_applications()
    )
    for wrong in (receipt, copy(actual), replace(actual, applications=())):
        with pytest.raises(ValueError, match="canonical compilation receipt"):
            restored.execution_outcome.require_function(wrong)
