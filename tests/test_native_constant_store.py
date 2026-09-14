"""Native entry stores preserve original producers, frames and source uses."""

import ast
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.captured_reference import CapturedReferenceKernel
from nominal_refactor_advisor.native_compilation import (
    ModuleNativeFrameOrigin,
    NativeCreationBackend,
    NativeEntryValueWindow,
    NativePythonCompilation,
    NativeConstantValue,
    SourceNativeFrameOrigin,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import CompilerStoredValue
from nominal_refactor_advisor.source_execution import (
    SourceCompilerStoredCapture,
    SourceLiteralCapture,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_documentation_store import execution


@pytest.mark.parametrize(
    "source",
    (
        '"""Summary.\n        Details.\n    """\n',
        'class Owner:\n    """Summary.\n        Details.\n    """\n',
        'class Owner:\n    """Summary.\n        Details.\n    """\n    global __doc__\n',
    ),
)
def test_receipt_retains_original_native_constant_store_and_frame(source):
    env = execution(source)
    doc = next(
        node for node in ast.walk(env.module.module) if isinstance(node, ast.Expr)
    )
    captured = env.capture_value(doc.value)
    assert isinstance(captured, SourceCompilerStoredCapture)
    receipt = captured.production
    compilation = env.module.native_compilation
    assert receipt is compilation.constant_store_for(
        SourceByteSpan.require_node(doc), "__doc__"
    )
    assert isinstance(receipt.value, NativeConstantValue)
    assert receipt.binding.value is receipt.value
    native_code = compilation.compile()
    if isinstance(env.module.module.body[0], ast.ClassDef):
        assert isinstance(receipt.frame, SourceNativeFrameOrigin)
        owner = env.module.module.body[0]
        assert receipt.frame.execution is env.class_entry(owner).capture.body
        native_code = next(
            value for value in native_code.co_consts if isinstance(value, CodeType)
        )
    else:
        assert isinstance(receipt.frame, ModuleNativeFrameOrigin)
        assert receipt.frame.compilation is compilation.identity
    instructions = tuple(dis.get_instructions(native_code))
    load_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.offset == receipt.value.instruction_offset
    )
    load, store = instructions[load_index : load_index + 2]
    assert store.offset == receipt.binding.instruction_offset
    assert store.opname == receipt.binding.operation.name
    assert store.argval == receipt.binding.name
    assert load.argval == captured.require_native_text()
    assert load.positions == store.positions


def test_constant_receipt_cache_pickles_without_executable_code(monkeypatch):
    source = 'class Owner:\n    """Summary.\n        Details.\n    """\n'
    node = ast.parse(source).body[0]
    doc_span = SourceByteSpan.require_node(node.body[0])
    compilation = NativePythonCompilation(source, "receipt.py")
    calls = []
    compile_original = NativePythonCompilation.compile

    def count(self):
        calls.append(self)
        return compile_original(self)

    monkeypatch.setattr(NativePythonCompilation, "compile", count)
    first = compilation.constant_store_for(doc_span, "__doc__")
    assert compilation.constant_store_for(doc_span, "__doc__") is first
    assert calls == [compilation]
    restored = pickle.loads(pickle.dumps(compilation))
    receipt = restored.constant_store_for(doc_span, "__doc__")
    assert (
        receipt.frame.execution
        is restored.class_capture_for(SourceByteSpan.require_node(node)).body
    )
    assert receipt.binding.value is receipt.value
    assert receipt.value.require_native_text() == first.value.require_native_text()
    assert len(calls) == 1


@pytest.mark.parametrize("duplicate", ("same_receipt", "distinct_receipt"))
def test_lookup_index_rejects_duplicate_rows_without_reusing_old_projection(duplicate):
    compilation = NativePythonCompilation('"documentation"\n', "duplicate.py")
    index = compilation._execution_outcome
    (original,) = index.constant_stores
    span, name = original.source_span, original.binding.name
    assert index.constant_store_for(span, name) is original
    second = original if duplicate == "same_receipt" else replace(original)
    changed = replace(
        index, scopes=(replace(index.scopes[0], constant_stores=(original, second)),)
    )
    with pytest.raises(ValueError, match="unique original receipt"):
        changed.constant_store_for(span, name)
    assert index.constant_store_for(span, name) is original


def test_lookup_index_keeps_binding_name_and_range_validity_in_the_join():
    compilation = NativePythonCompilation('"documentation"\n', "lookup.py")
    index = compilation._execution_outcome
    (original,) = index.constant_stores
    other = replace(original, binding=replace(original.binding, name="other"))
    changed = replace(
        index, scopes=(replace(index.scopes[0], constant_stores=(original, other)),)
    )
    assert (
        changed.constant_store_for(original.source_span, original.binding.name)
        is original
    )
    assert changed.constant_store_for(other.source_span, "other") is other
    with pytest.raises(ValueError):
        changed.constant_store_for(original.source_span, "absent")
    with pytest.raises(ValueError):
        replace(changed, has_incomplete_ranges=True).constant_store_for(
            other.source_span, "other"
        )


def test_copied_read_marker_context_and_foreign_execution_cannot_borrow_receipt():
    env = execution('"documentation"\n')
    node = env.module.module.body[0].value
    captured = env.capture_value(node)
    read = captured.read
    assert isinstance(read.use.value, CompilerStoredValue)
    for changed in (
        replace(read, use=replace(read.use)),
        replace(read, use=replace(read.use, value=replace(read.use.value))),
        replace(read, context=replace(read.context)),
    ):
        with pytest.raises(ValueError):
            SourceCompilerStoredCapture(env, changed).require_native_text()
    foreign = execution('"documentation"\n')
    with pytest.raises(ValueError):
        SourceCompilerStoredCapture(foreign, read).require_native_text()
    with pytest.raises(ValueError):
        env.kernel._compiler_stored_value_resolution(
            replace(read.use.value), (read, frozenset())
        )


def test_plain_literal_leaf_cannot_reinterpret_compiler_marked_value():
    env = execution('"""Summary.\n        Details.\n    """\n')
    captured = env.capture_value(env.module.module.body[0].value)
    with pytest.raises(ValueError):
        SourceLiteralCapture(env, captured.read).require_native_text()


def test_generic_kernel_has_no_compiler_value_authority():
    env = execution('"documentation"\n')
    read = env.source.value_reads_by_node[env.module.module.body[0].value]
    generic = CapturedReferenceKernel(env.initial, env)
    with pytest.raises(ValueError):
        generic.read(read).require_closed()


def test_unsupported_native_backend_never_falls_back_to_raw_literal(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    env = execution('"documentation"\n')
    with pytest.raises(ValueError):
        env.capture_value(env.module.module.body[0].value).require_native_text()


@pytest.mark.parametrize("disturbance", ("jump", "extra_value", "unsupported"))
def test_entry_window_does_not_skip_an_intervening_operation(disturbance):
    compilation = NativePythonCompilation('"documentation"\n', "window.py")
    code = compilation.compile()
    instructions = tuple(dis.get_instructions(code))
    load = next(
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_CONST" and instruction.argval == "documentation"
    )
    store = next(
        instruction
        for instruction in instructions
        if instruction.opname == "STORE_NAME"
    )
    window = NativeEntryValueWindow(
        code, NativeCreationBackend.current().primitive_operations
    )
    window.observe(load)
    branch_code = compile(
        "if flag:\n    value = 'first'\nelse:\n    value = 'second'\n",
        "branch.py",
        "exec",
    )
    jump_target = next(
        instruction
        for instruction in dis.get_instructions(branch_code)
        if instruction.is_jump_target
    )
    changed = {
        "jump": jump_target,
        "extra_value": load,
        "unsupported": store._replace(opcode=-1),
    }[disturbance]
    window.observe(changed)
    window.observe(store)
    assert window.closed
    assert window.receipt(ModuleNativeFrameOrigin(compilation.identity)) is None


def test_entry_stores_and_function_returns_share_one_walk_per_emitted_scope(
    monkeypatch,
):
    source = '"module documentation"\n' + "\n".join(
        f'def leaf{index}():\n    "function metadata"\n    return {index}'
        for index in range(100)
    )
    backend = NativeCreationBackend.current()
    original = type(backend).instructions
    observed = []

    def count(self, code):
        observed.append(code.co_qualname)
        yield from original(self, code)

    monkeypatch.setattr(type(backend), "instructions", count)
    compilation = NativePythonCompilation(source, "leafs.py")
    nodes = ast.parse(source).body
    documentation = compilation.constant_store_for(
        SourceByteSpan.require_node(nodes[0]), "__doc__"
    )
    assert documentation.value.require_native_text() == "module documentation"
    # Function docstrings are code metadata, not executable __doc__ stores.
    assert compilation.execution_outcome.constant_stores == (documentation,)
    assert len(observed) == 101
    assert set(observed) == {"<module>", *(f"leaf{index}" for index in range(100))}
    for index, node in enumerate(nodes[1:]):
        execution = compilation.execution_for(SourceByteSpan.require_node(node))
        receipt = compilation.return_from(execution)
        assert receipt.frame.is_body_of(execution)
        assert receipt.value.require_native_scalar() == index
    assert len(observed) == 101
