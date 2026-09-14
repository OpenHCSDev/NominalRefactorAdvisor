"""Actual compiler prologue slots, without inferred class activation safety."""

import ast
import builtins
from dataclasses import fields, is_dataclass, replace
import dis
import pickle
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.native_compilation import (
    ExactNativeClassCapture,
    ExactNativeClassPrologue,
    NativeCodeObservation,
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativeBindingTransfer,
    NativeBindingTransferResolverABC,
    NativePrimitiveOperation,
    NativeTypedValue,
    NativePythonCompilation,
    OpenNativeClassCapture,
    OpenNativeClassPrologue,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _case(monkeypatch, source, **extra):
    compilation = NativePythonCompilation(source, "native_class_prologue_case.py")
    code = compilation.compile()
    native_compile = NativePythonCompilation.compile
    monkeypatch.setattr(
        NativePythonCompilation,
        "compile",
        lambda self: code if self is compilation else native_compile(self),
    )
    span = next(
        SourceByteSpan.require_node(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
    )
    capture = compilation.class_capture_for(span)
    assert isinstance(capture, ExactNativeClassCapture)
    assert isinstance(capture.prologue, ExactNativeClassPrologue)
    invocations = []
    snapshots = []

    def builder(body, name, *bases, **keywords):
        invocations.append(body.__code__)
        return builtins.__build_class__(body, name, *bases, **keywords)

    def trace(frame, event, arg):
        if any(frame.f_code is body for body in invocations):
            frame.f_trace_opcodes = True
            if (
                event == "opcode"
                and frame.f_lasti == capture.prologue.body_instruction_offset
            ):
                snapshots.append((dict(frame.f_locals), dict(frame.f_globals)))
        return trace

    namespace = {
        "__name__": "native_class_prologue_case",
        "__builtins__": dict(vars(builtins), __build_class__=builder),
        "record": lambda: None,
        **extra,
    }
    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        exec(code, namespace)
    finally:
        sys.settrace(previous_trace)
    assert len(snapshots) == 1
    return compilation, capture, invocations[0], snapshots[0], namespace


@pytest.mark.parametrize(
    "source",
    (
        "class Target:\n    record()\n    later = 1\n",
        "class Target:\n    record()\n    later: int = 1\n",
        "from __future__ import annotations\nclass Target:\n    record()\n    later: int = 1\n",
        "class Target:\n    'Source docstring'\n    record()\n",
        "class Target:\n    global __module__, __qualname__\n    record()\n",
        "class Target:\n    record()\n    builtins = 1\n",
        "class Target:\n    record()\n    def method(self): self.value = 1\n",
        "def outer():\n    enclosing = 3\n    class Target:\n        record()\n        saved = enclosing\n    return Target\nTarget = outer()\n",
        "def outer():\n    __module__ = __qualname__ = None\n    class Target:\n        nonlocal __module__, __qualname__\n        record()\n    return Target\nTarget = outer()\n",
    ),
)
def test_prologue_matches_actual_native_entry_slots(monkeypatch, source):
    _, capture, body, (locals_at_cut, globals_at_cut), _ = _case(monkeypatch, source)
    prologue = capture.prologue
    local_bindings = tuple(
        b
        for b in prologue.bindings
        if b.operation
        in {
            NativePrimitiveOperation.STORE_NAME,
            NativePrimitiveOperation.SETUP_ANNOTATIONS,
        }
    )
    assert set(locals_at_cut) == {binding.name for binding in local_bindings}
    assert "later" not in locals_at_cut
    instructions = {
        instruction.offset: instruction for instruction in dis.get_instructions(body)
    }
    for binding in prologue.bindings:
        instruction = instructions[binding.instruction_offset]
        assert binding.instruction_offset < prologue.body_instruction_offset
        assert instruction.opname == binding.operation.name
        if binding.operation is NativePrimitiveOperation.STORE_GLOBAL:
            assert instruction.opname == "STORE_GLOBAL"
            assert binding.name in globals_at_cut
            assert binding.name not in locals_at_cut
        elif binding.operation is NativePrimitiveOperation.STORE_DEREF:
            assert instruction.opname == "STORE_DEREF"
        elif binding.operation is NativePrimitiveOperation.MAKE_CELL:
            assert binding.name in body.co_cellvars
            assert binding.name not in body.co_freevars
        else:
            assert instruction.opname in {"STORE_NAME", "SETUP_ANNOTATIONS"}
        if instruction.opname != "SETUP_ANNOTATIONS":
            assert binding.name == instruction.argval
        assert binding.operand_index == instruction.arg
        if instruction.opname in {"STORE_NAME", "SETUP_ANNOTATIONS"}:
            value = binding.value
            prologue.require_value(value)
            if isinstance(value, NativeTypedValue):
                assert type(locals_at_cut[binding.name]) is value.native_type


@pytest.mark.parametrize(
    "source",
    (
        "class Target: pass\n",
        "class Target(object): pass\n",
        "class Target:\n    pass\n",
        "class Target: global __module__; pass\n",
        "def outer():\n    class Target: pass\n    return Target\nTarget = outer()\n",
    ),
)
def test_pass_only_class_has_actual_native_prefix_cut(monkeypatch, source):
    _, capture, body, (locals_at_cut, globals_at_cut), namespace = _case(
        monkeypatch, source
    )
    instructions = tuple(dis.get_instructions(body))
    cut = next(
        i for i in instructions if i.offset == capture.prologue.body_instruction_offset
    )
    assert cut.offset > max(b.instruction_offset for b in capture.prologue.bindings)
    for binding in capture.prologue.bindings:
        if binding.operation is NativePrimitiveOperation.STORE_GLOBAL:
            assert binding.name in globals_at_cut
            assert binding.name not in locals_at_cut
        else:
            assert binding.name in locals_at_cut
    assert isinstance(namespace["Target"], type)
    prefix = NativeCodeObservation(body)
    for instruction in instructions:
        prefix.observe(instruction)
    if prefix.boundary is None:
        # Native3.11's one-line pass has no distinct source body range at all.
        # Exactness comes from the actual native terminal primitive, not its line.
        assert (
            NativeCreationBackend.current()
            .primitive_operations[cut.opcode]
            .terminates_prefix
        )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Native generic-class syntax")
def test_generic_inline_pass_retains_actual_native_prefix(monkeypatch):
    _, capture, _, (locals_at_cut, _), _ = _case(monkeypatch, "class Target[T]: pass\n")
    assert {b.name for b in capture.prologue.bindings} <= set(locals_at_cut)


@pytest.mark.parametrize(
    "source",
    (
        "class Target: __module__ = 'changed'; later = 1\n",
        "class Target: __qualname__ = 'changed'; later = 1\n",
        "class Target: later = 1; pass\n",
    ),
)
def test_same_line_source_writes_are_not_initial_namespace(monkeypatch, source):
    _, capture, body, (locals_at_cut, _), namespace = _case(monkeypatch, source)
    assert "later" not in locals_at_cut
    assert namespace["Target"].later == 1
    assert locals_at_cut["__module__"] == "native_class_prologue_case"
    assert locals_at_cut["__qualname__"] == "Target"
    prefix_offsets = {b.instruction_offset for b in capture.prologue.bindings}
    explicit_stores = tuple(
        i
        for i in dis.get_instructions(body)
        if i.opname == "STORE_NAME" and i.positions.col_offset
    )
    assert explicit_stores
    assert prefix_offsets.isdisjoint(i.offset for i in explicit_stores)


def test_missing_terminal_and_geometric_boundary_stays_open(monkeypatch):
    _, _, body, _, _ = _case(monkeypatch, "class Target:\n    record()\n")
    prefix = NativeCodeObservation(body)
    for instruction in dis.get_instructions(body):
        header = (body.co_firstlineno, body.co_firstlineno, 0, 0)
        if tuple(instruction.positions) != header and None not in instruction.positions:
            break
        prefix.observe(instruction)
    result = NativeCreationBackend.current().class_prologue(prefix)
    assert isinstance(result, OpenNativeClassPrologue)
    assert result.reason is NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES


def test_make_cell_retains_actual_owned_slot_without_namespace_entry(monkeypatch):
    _, capture, body, (locals_at_cut, _), _ = _case(
        monkeypatch,
        "class Target:\n    def method(self): return __class__\n",
    )
    creations = tuple(
        b
        for b in capture.prologue.bindings
        if b.operation is NativePrimitiveOperation.MAKE_CELL
    )
    assert creations
    instructions = {i.offset: i for i in dis.get_instructions(body)}
    for creation in creations:
        native = instructions[creation.instruction_offset]
        assert native.opname == "MAKE_CELL"
        assert native.arg == creation.operand_index
        assert native.argval == creation.name
        assert creation.name in body.co_cellvars
        assert creation.name not in body.co_freevars
        assert creation.name not in locals_at_cut


def test_one_creation_receipt_does_not_identify_repeated_runtime_cells():
    source = (
        "def make():\n"
        "    class Target:\n"
        "        def owner(self): return __class__\n"
        "    return Target\n"
        "first = make()\nsecond = make()\n"
    )
    compilation = NativePythonCompilation(source, "repeated_cells.py")
    class_node = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)
    )
    capture = compilation.class_capture_for(SourceByteSpan.require_node(class_node))
    assert isinstance(capture, ExactNativeClassCapture)
    assert isinstance(capture.prologue, ExactNativeClassPrologue)
    assert any(
        b.operation is NativePrimitiveOperation.MAKE_CELL
        for b in capture.prologue.bindings
    )
    namespace = {}
    exec(compilation.compile(), namespace)
    first, second = namespace["first"], namespace["second"]
    (first_cell,) = first.owner.__closure__
    (second_cell,) = second.owner.__closure__
    assert first_cell is not second_cell
    assert first_cell.cell_contents is first
    assert second_cell.cell_contents is second


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred class annotations"
)
@pytest.mark.parametrize(
    "body,in_prefix",
    (
        ("field: record()", True),
        ("field: record()\npass", False),
        ("field: record() = value()", False),
    ),
)
def test_annotation_helper_creation_is_not_annotation_body_execution(
    monkeypatch, body, in_prefix
):
    selected = []
    native_require = NativeCodeObservation.require_creation

    def observed_require(prefix, instruction):
        emission = native_require(prefix, instruction)
        selected.append(emission)
        return emission

    monkeypatch.setattr(NativeCodeObservation, "require_creation", observed_require)
    calls = []
    source = "class Target:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    _, capture, raw_body, (locals_at_cut, _), namespace = _case(
        monkeypatch,
        source,
        record=lambda: calls.append("annotation") or int,
        value=lambda: calls.append("value") or 17,
    )
    assert calls == (["value"] if "value()" in body else [])
    helper = namespace["Target"].__annotate__
    assert any(value is helper for value in locals_at_cut.values()) is in_prefix
    assert bool(selected) is in_prefix
    assert all(emission.code is helper.__code__ for emission in selected)
    assert all(emission.containing_code is raw_body for emission in selected)
    assert "field" not in {binding.name for binding in capture.prologue.bindings}
    assert namespace["Target"].__annotations__["field"] is int
    assert calls[-1] == "annotation"
    assert calls.count("annotation") == 1


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred class annotations"
)
@pytest.mark.parametrize(
    "damage",
    (
        "cloned_instruction",
        "foreign_code",
        "foreign_attachment",
        "unsupported_flag",
        "missing_range",
    ),
)
def test_unobserved_annotation_creation_provenance_remains_open(monkeypatch, damage):
    compilation = NativePythonCompilation(
        "class Target:\n    field: int\n", "helper_provenance.py"
    )
    code = compilation.compile()
    backend = NativeCreationBackend.current()
    inventory = backend.inventory(code, compilation.identity)
    (prefix,) = tuple(
        prefix
        for prefix in inventory.prefixes.values()
        if prefix.emissions and prefix.code is not code and prefix.boundary is None
    )
    (emission,) = prefix.emissions
    (attachment,) = emission.attachments
    assert backend.class_prologue(prefix).__class__ is ExactNativeClassPrologue
    if damage == "cloned_instruction":
        with pytest.raises(ValueError, match="outside this prefix"):
            prefix.require_creation(attachment._replace())
        return
    if damage == "foreign_code":
        prefix.emissions[:] = [replace(emission, containing_code=prefix.code.replace())]
    elif damage == "foreign_attachment":
        prefix.emissions[:] = [replace(emission, attachments=[attachment._replace()])]
    else:
        native_instructions = backend.instructions

        def corrupted_instructions(parent):
            for instruction in native_instructions(parent):
                if parent is prefix.code and instruction.offset == attachment.offset:
                    yield instruction._replace(
                        **(
                            {"arg": 1 << 30}
                            if damage == "unsupported_flag"
                            else {"positions": dis.Positions(None, None, None, None)}
                        )
                    )
                else:
                    yield instruction

        monkeypatch.setattr(backend, "instructions", corrupted_instructions)
        prefix = backend.inventory(code, compilation.identity).prefixes[id(prefix.code)]
    assert isinstance(backend.class_prologue(prefix), OpenNativeClassPrologue)


def test_captured_outer_cell_store_is_not_fresh_despite_other_owned_cells(monkeypatch):
    _, capture, body, _, _ = _case(
        monkeypatch,
        "def outer():\n"
        "    __module__ = 'outer'\n"
        "    class Target:\n"
        "        nonlocal __module__\n"
        "        def method(self): return __class__\n"
        "    return Target\n"
        "Target = outer()\n",
    )
    prologue = capture.prologue
    creations = tuple(
        b
        for b in prologue.bindings
        if b.operation is NativePrimitiveOperation.MAKE_CELL
    )
    stores = tuple(
        b
        for b in prologue.bindings
        if b.operation is NativePrimitiveOperation.STORE_DEREF
        and b.name in body.co_freevars
    )
    assert creations and stores
    for store in stores:
        assert store.name not in body.co_cellvars
        assert all(c.operand_index != store.operand_index for c in creations)
        with pytest.raises(ValueError, match="no unique prior creation"):
            prologue.require_fresh_cell_store(store)


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native class-dictionary cell store"
)
def test_fresh_cell_store_joins_exact_prior_native_slot(monkeypatch):
    _, capture, body, (locals_at_cut, _), _ = _case(
        monkeypatch, "class Target:\n    def method(self): pass\n"
    )
    prologue = capture.prologue
    stores = tuple(
        b
        for b in prologue.bindings
        if b.operation is NativePrimitiveOperation.STORE_DEREF
    )
    assert stores
    for store in stores:
        assert store.name in body.co_cellvars
        assert store.name not in body.co_freevars
        assert store.name not in locals_at_cut
        prologue.require_fresh_cell_store(store)
        (creation,) = tuple(
            b
            for b in prologue.bindings
            if b.operation is NativePrimitiveOperation.MAKE_CELL
            and b.operand_index == store.operand_index
        )
        instructions = {i.offset: i for i in dis.get_instructions(body)}
        assert (
            instructions[creation.instruction_offset].arg
            == instructions[store.instruction_offset].arg
        )
        with pytest.raises(ValueError, match="actual cell store"):
            prologue.require_fresh_cell_store(replace(store))
        for damaged_creation in (
            replace(creation, operand_index=creation.operand_index + 1),
            replace(creation, instruction_offset=store.instruction_offset + 1),
        ):
            damaged = replace(
                prologue,
                bindings=tuple(
                    damaged_creation if b is creation else b for b in prologue.bindings
                ),
            )
            with pytest.raises(ValueError, match="no unique prior creation"):
                damaged.require_fresh_cell_store(store)
        duplicate = replace(prologue, bindings=(*prologue.bindings, creation))
        with pytest.raises(ValueError, match="no unique prior creation"):
            duplicate.require_fresh_cell_store(store)
        unresolved = OpenNativeClassPrologue(
            NativeExecutionUnavailable.UNSUPPORTED_PROLOGUE
        )
        with pytest.raises(ValueError, match="unresolved"):
            unresolved.require_fresh_cell_store(store)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Native generic-class syntax")
@pytest.mark.parametrize("annotation", ("", ": T"))
def test_generic_native_slots_preserve_actual_creator_origin(monkeypatch, annotation):
    _, capture, _, (locals_at_cut, _), _ = _case(
        monkeypatch, f"class Target[T]:\n    record()\n    later{annotation} = 1\n"
    )
    assert set(locals_at_cut) == {
        binding.name
        for binding in capture.prologue.bindings
        if binding.operation
        in {
            NativePrimitiveOperation.STORE_NAME,
            NativePrimitiveOperation.SETUP_ANNOTATIONS,
        }
    }
    assert capture.builder.frame is capture.creation.frame
    assert "__type_params__" in locals_at_cut


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred annotation tail"
)
def test_late_header_positioned_stores_never_reenter_initial_prefix(monkeypatch):
    _, capture, body, _, _ = _case(
        monkeypatch, "class Target:\n    record()\n    later: int = 1\n"
    )
    header = (body.co_firstlineno, body.co_firstlineno, 0, 0)
    tail = tuple(
        i
        for i in dis.get_instructions(body)
        if i.offset > capture.prologue.body_instruction_offset
        and tuple(i.positions) == header
        and i.opname == "STORE_NAME"
    )
    assert tail, "This native control must actually contain colliding tail spans"
    assert not {i.offset for i in tail}.intersection(
        binding.instruction_offset for binding in capture.prologue.bindings
    )


def test_read_before_later_local_has_no_initial_binding(monkeypatch):
    _, capture, _, _, namespace = _case(
        monkeypatch,
        "import builtins\nclass Target:\n    saved = builtins.property\n    builtins = 1\n",
    )
    assert namespace["Target"].saved is builtins.property
    assert "builtins" not in {binding.name for binding in capture.prologue.bindings}


@pytest.mark.parametrize(
    "source",
    (
        "class Target:\n    record()\n",
        "class Target:\n    field: int\n",
    ),
)
def test_compact_capture_and_queried_compilation_pickle_without_native_code(
    monkeypatch,
    source,
):
    compilation, capture, _, _, _ = _case(monkeypatch, source)
    restored = pickle.loads(pickle.dumps(compilation))
    assert restored.class_capture_for(capture.source_span) == capture
    pending = [restored]
    while pending:
        value = pending.pop()
        assert not isinstance(value, (ast.AST, CodeType, dis.Instruction))
        if is_dataclass(value):
            pending.extend(getattr(value, field.name) for field in fields(value))
            pending.extend(vars(value).values())
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)


def test_unadmitted_backend_cannot_project_prologue(monkeypatch):
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    compilation = NativePythonCompilation("class Target: pass\n", "unsupported.py")
    capture = compilation.class_capture_for(
        SourceByteSpan.require_node(ast.parse(compilation.source).body[0])
    )
    assert isinstance(capture, OpenNativeClassCapture)
    assert capture.reason is NativeExecutionUnavailable.UNSUPPORTED_COMPILER
    prefix = NativeCodeObservation(compilation.compile())
    result = SpanOnlyCreationBackend().class_prologue(prefix)
    assert isinstance(result, OpenNativeClassPrologue)
    assert result.reason is NativeExecutionUnavailable.UNSUPPORTED_COMPILER


@pytest.mark.parametrize(
    "damage", ("unknown", "jump", "missing_range", "malformed_operand")
)
def test_unproved_native_prefix_remains_open(monkeypatch, damage):
    _, _, body, _, _ = _case(monkeypatch, "class Target:\n    record()\n")
    prefix = NativeCodeObservation(body)
    for instruction in dis.get_instructions(body):
        prefix.observe(instruction)
    store_index = next(
        i
        for i, instruction in enumerate(prefix.instructions)
        if instruction.opname == "STORE_NAME"
    )
    instruction = prefix.instructions[store_index]
    if damage == "unknown":
        # NOP is an admitted inert primitive; raising remains unsupported here.
        instruction = instruction._replace(
            opcode=dis.opmap["RAISE_VARARGS"], opname="RAISE_VARARGS"
        )
    elif damage == "jump":
        instruction = instruction._replace(
            **(
                {"is_jump_target": True}
                if "is_jump_target" in instruction._fields
                else {"label": 1}
            )
        )
    elif damage == "missing_range":
        instruction = instruction._replace(
            positions=dis.Positions(None, None, None, None)
        )
    else:
        instruction = instruction._replace(argval=None)
    prefix._instructions[id(prefix.instructions[store_index])] = instruction
    result = NativeCreationBackend.current().class_prologue(prefix)
    assert isinstance(result, OpenNativeClassPrologue)


def test_prefix_does_not_restart_after_first_body_instruction(monkeypatch):
    _, capture, body, _, _ = _case(monkeypatch, "class Target:\n    record()\n")
    prefix = NativeCodeObservation(body)
    instructions = tuple(dis.get_instructions(body))
    for instruction in instructions:
        prefix.observe(instruction)
    original = prefix.initial_instructions
    prefix.observe(instructions[0])
    assert prefix.initial_instructions == original
    assert prefix.boundary.offset == capture.prologue.body_instruction_offset


def test_native_ensure_keeps_existing_value_while_store_replaces_it(monkeypatch):
    annotations = {"existing": object()}
    replaced_module = object()

    class Prepared(type):
        @classmethod
        def __prepare__(cls, name, bases):
            return {"__annotations__": annotations, "__module__": replaced_module}

    _, capture, _, (locals_at_cut, _), namespace = _case(
        monkeypatch,
        "from __future__ import annotations\nclass Target(metaclass=Prepared):\n    record()\n    later: int\n",
        Prepared=Prepared,
    )
    by_name = {binding.name: binding for binding in capture.prologue.bindings}
    assert (
        by_name["__annotations__"].operation
        is NativePrimitiveOperation.SETUP_ANNOTATIONS
    )
    assert by_name["__module__"].operation is NativePrimitiveOperation.STORE_NAME
    assert locals_at_cut["__annotations__"] is annotations
    assert namespace["Target"].__annotations__ is annotations
    assert locals_at_cut["__module__"] == "native_class_prologue_case"
    assert locals_at_cut["__module__"] is not replaced_module


class _BindingResolver(NativeBindingTransferResolverABC):
    def _cell_creation_resolution(self, binding):
        return "cell_creation", binding

    def _local_store_resolution(self, binding):
        return "local_store", binding

    def _global_store_resolution(self, binding):
        return "global_store", binding

    def _cell_store_resolution(self, binding):
        return "cell_store", binding

    def _local_ensure_resolution(self, binding):
        return "local_ensure", binding


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        (NativePrimitiveOperation.STORE_NAME, "local_store"),
        (NativePrimitiveOperation.STORE_GLOBAL, "global_store"),
        (NativePrimitiveOperation.STORE_DEREF, "cell_store"),
        (NativePrimitiveOperation.MAKE_CELL, "cell_creation"),
        (NativePrimitiveOperation.SETUP_ANNOTATIONS, "local_ensure"),
    ),
)
def test_primitive_owns_actual_resolution_without_destination_mirror(
    operation, expected
):
    binding = NativeBindingTransfer(operation, "selected", 12, 0)
    result, retained_binding = binding.resolve(_BindingResolver())
    assert result == expected
    assert retained_binding is binding
    restored = pickle.loads(pickle.dumps(binding))
    assert restored == binding
    assert restored.operation is operation


def test_all_native_primitive_members_pickle_without_serializing_callbacks():
    for operation in NativePrimitiveOperation:
        assert pickle.loads(pickle.dumps(operation)) is operation
