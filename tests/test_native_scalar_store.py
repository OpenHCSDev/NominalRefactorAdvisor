"""Adjacent native stores retain distinct producer/target spans and original frames."""

import ast
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    NativeInstructionSite,
    NativePythonCompilation,
    NativePrimitiveOperation,
    NativeValueStoreStream,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def assignment(source):
    tree = ast.parse(source)
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign))


def store_for(compilation, node):
    return compilation.scalar_store_for(
        SourceByteSpan.require_node(node.value),
        SourceByteSpan.require_node(node.targets[0]),
        node.targets[0].id,
    )


@pytest.mark.parametrize("value", ("None", "True", "7", "'label'"))
@pytest.mark.parametrize("in_class", (False, True))
def test_scalar_pair_matches_actual_native_instructions(value, in_class):
    source = ("class Family:\n    " if in_class else "") + "key = " + value + "\n"
    compilation = NativePythonCompilation(source, "scalar.py")
    node = assignment(source)
    receipt = store_for(compilation, node)
    assert receipt.production_span != receipt.source_span
    assert receipt.value.require_native_scalar() == ast.literal_eval(node.value)
    assert receipt.binding.value is receipt.value
    assert receipt.binding.operation is NativePrimitiveOperation.STORE_NAME
    code = compilation.compile()
    if in_class:
        class_node = ast.parse(source).body[0]
        assert (
            receipt.frame.execution
            is compilation.class_capture_for(
                SourceByteSpan.require_node(class_node)
            ).body
        )
        code = next(value for value in code.co_consts if isinstance(value, CodeType))
    else:
        assert receipt.frame.compilation is compilation.identity
    instructions = tuple(dis.get_instructions(code))
    index = next(
        i
        for i, instruction in enumerate(instructions)
        if instruction.offset == receipt.value.instruction_offset
    )
    production, store = instructions[index : index + 2]
    assert store.offset == receipt.binding.instruction_offset
    assert (
        NativeInstructionSite(code, production).source_span == receipt.production_span
    )
    assert NativeInstructionSite(code, store).source_span == receipt.source_span


def test_later_compiler_overwrite_cannot_substitute_for_source_assignment():
    source = "class Family:\n    __static_attributes__ = None\n"
    compilation = NativePythonCompilation(source, "overwrite.py")
    receipt = store_for(compilation, assignment(source))
    assert receipt.value.require_native_scalar() is None
    assert receipt.production_span != receipt.source_span


def test_scalar_receipts_do_not_broaden_documentation_entry_contract():
    source = "first = 1\ntext = 'later'\n"
    compilation = NativePythonCompilation(source, "entry.py")
    for node in ast.parse(source).body:
        assert store_for(compilation, node).binding.name == node.targets[0].id
        with pytest.raises(ValueError):
            compilation.constant_store_for(
                SourceByteSpan.require_node(node.value), node.targets[0].id
            )


@pytest.mark.parametrize(
    "source",
    (
        "one = two = None\n",
        "key = unknown\n",
        "key = unknown()\n",
        "key = 3.25\n",
        "key = b'label'\n",
        "key = 2j\n",
    ),
)
def test_non_immediate_or_non_scalar_binding_is_not_inferred(source):
    with pytest.raises(ValueError, match="Native (value store|scalar production)"):
        store_for(NativePythonCompilation(source, "reject.py"), assignment(source))


@pytest.mark.parametrize(
    "disturbance", ("intervening", "producer-entry", "store-entry")
)
def test_instruction_pair_rejects_intervening_work_or_jump_entry(disturbance):
    backend = NativeCreationBackend.current()
    code = compile("key = None\n", "window.py", "exec")
    instructions = tuple(backend.instructions(code))
    store_index = next(
        i
        for i, instruction in enumerate(instructions)
        if instruction.opname == "STORE_NAME"
    )
    production, store = instructions[store_index - 1 : store_index + 1]
    if disturbance == "intervening":
        selected = (production, instructions[0], store)
    else:
        marked = production if disturbance == "producer-entry" else store
        marked = (
            marked._replace(label=0)
            if hasattr(marked, "label")
            else marked._replace(is_jump_target=True)
        )
        selected = (
            (marked, store) if disturbance == "producer-entry" else (production, marked)
        )
    stream = NativeValueStoreStream(code, backend.primitive_operations)
    for instruction in selected:
        stream.observe(instruction)
    assert not any(segment.stores for segment in stream.segments)


def test_duplicate_receipts_and_stale_index_cannot_supply_unique_store():
    source = "key = None\n"
    compilation = NativePythonCompilation(source, "duplicate.py")
    node = assignment(source)
    original = store_for(compilation, node)
    index = compilation._execution_outcome
    for other in (original, replace(original)):
        duplicated = replace(
            index, scopes=(replace(index.scopes[0], value_stores=(original, other)),)
        )
        with pytest.raises(ValueError, match="unique original receipt"):
            duplicated.scalar_store_for(
                original.production_span, original.source_span, original.binding.name
            )
    assert store_for(compilation, node) is original


def test_pickle_preserves_canonical_frames_without_executable_code(monkeypatch):
    source = "class Family:\n    key = None\n"
    original = NativePythonCompilation(source, "pickle.py")
    receipt = store_for(original, assignment(source))
    payload = pickle.dumps(original)

    def no_compile(self):
        raise AssertionError("Restored compact receipts must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    store = store_for(restored, assignment(source))
    assert store is not receipt
    assert (
        store.frame.execution
        is restored.class_capture_for(
            SourceByteSpan.require_node(ast.parse(source).body[0])
        ).body
    )
    assert store.value is store.binding.value


def test_unsupported_compiler_does_not_infer_scalar_store(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    source = "key = None\n"
    with pytest.raises(ValueError):
        store_for(NativePythonCompilation(source, "unsupported.py"), assignment(source))
