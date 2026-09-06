"""Native creation-use evidence selects raw bodies, not decorated bindings."""

import ast
from dataclasses import dataclass, fields, is_dataclass
import dis
import sys
from types import CodeType, FunctionType

import pytest

from nominal_refactor_advisor.native_compilation import (
    CPython314CreationBackend,
    ExactNativeFunctionExecution,
    NativeCodeEmission,
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativeFunctionExecution,
    NativeFunctionExecutionMode,
    NativePythonCompilation,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class NativeCase:
    compilation: NativePythonCompilation
    execution: NativeFunctionExecution
    module_code: CodeType
    function: FunctionType
    namespace: dict[str, object]
    span: SourceByteSpan
    selected_emissions: tuple[NativeCodeEmission, ...]


def _native_case(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    *,
    target_name: str = "sample",
    namespace: dict[str, object] | None = None,
) -> NativeCase:
    definitions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == target_name
    ]
    (definition,) = definitions
    span = SourceByteSpan.require_node(definition)
    native_compile = NativePythonCompilation.compile
    native_select = NativeCreationBackend.select
    compiled: list[CodeType] = []
    captured: list[FunctionType] = []
    selected_emissions: list[NativeCodeEmission] = []

    def observed_compile(owner: NativePythonCompilation) -> CodeType:
        code = native_compile(owner)
        compiled.append(code)
        return code

    def capture(function: FunctionType) -> FunctionType:
        captured.append(function)
        return function

    def observed_select(
        backend: NativeCreationBackend, bucket: list[NativeCodeEmission]
    ) -> NativeCodeEmission | None:
        selected = native_select(backend, bucket)
        if selected is not None:
            selected_emissions.append(selected)
        return selected

    monkeypatch.setattr(NativePythonCompilation, "compile", observed_compile)
    monkeypatch.setattr(NativeCreationBackend, "select", observed_select)
    compilation = NativePythonCompilation(source, "native_creation_case.py")
    execution = compilation.execution_for(span)
    assert captured == []  # Static evidence acquisition never executes source.
    (module_code,) = compiled
    namespace = {} if namespace is None else dict(namespace)
    namespace["capture"] = capture
    # Only this controlled test source is executed, using the SAME native code
    # produced for the receipt, so no second compilation can change the oracle.
    exec(module_code, namespace)
    (function,) = captured
    return NativeCase(
        compilation,
        execution,
        module_code,
        function,
        namespace,
        span,
        tuple(selected_emissions),
    )


def _assert_matches_native(case: NativeCase) -> None:
    execution = case.execution
    assert isinstance(execution, ExactNativeFunctionExecution)
    assert execution.native_flags == case.function.__code__.co_flags
    assert execution.mode is NativeFunctionExecutionMode.from_flags(
        case.function.__code__.co_flags
    )
    assert execution.source_span == case.span
    assert execution.compilation is case.compilation.identity
    assert execution.violation is None
    assert case.compilation.execution_for(case.span) is execution
    (selected,) = tuple(
        emission
        for emission in case.selected_emissions
        if emission.receipt is execution
    )
    # Check the selected native object before its compact projection. Helpers
    # can have identical flags, names, signatures and spans to the actual body.
    assert selected.code is case.function.__code__


@pytest.mark.parametrize(
    "declaration",
    (
        pytest.param(
            "def sample(value: int) -> int:\n    return value\n", id="ordinary"
        ),
        pytest.param("def sample(value: int):\n    yield value\n", id="generator"),
        pytest.param(
            "async def sample(value: int) -> int:\n    return value\n", id="coroutine"
        ),
        pytest.param(
            "async def sample(value: int):\n    yield value\n", id="async-generator"
        ),
    ),
)
@pytest.mark.parametrize(
    "future", (False, True), ids=("native-annotations", "future-annotations")
)
def test_annotated_body_matches_actual_compiled_function(
    monkeypatch: pytest.MonkeyPatch, declaration: str, future: bool
) -> None:
    prefix = "from __future__ import annotations\n" if future else ""
    _assert_matches_native(
        _native_case(monkeypatch, prefix + "@capture\n" + declaration)
    )


def test_closure_and_both_default_families_preserve_creation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _native_case(
        monkeypatch,
        "def outer(seed):\n"
        "    @capture\n"
        "    def sample(value: int = seed, *, extra: int = seed) -> int:\n"
        "        return value + extra + seed\n"
        "    return sample\n"
        "sample = outer(7)\n",
    )
    assert case.function.__closure__ is not None
    assert case.function.__defaults__ == (7,)
    assert case.function.__kwdefaults__ == {"extra": 7}
    _assert_matches_native(case)


@pytest.mark.parametrize("choose", (False, True), ids=("else-entry", "if-entry"))
def test_native_conditional_entries_keep_their_creation_boundaries(
    monkeypatch: pytest.MonkeyPatch, choose: bool
) -> None:
    source = (
        "if choose:\n"
        "    @capture\n"
        "    def first(value: int) -> int:\n"
        "        return value\n"
        "else:\n"
        "    @capture\n"
        "    async def second(value: int) -> int:\n"
        "        return value\n"
    )
    _assert_matches_native(
        _native_case(
            monkeypatch,
            source,
            target_name="first" if choose else "second",
            namespace={"choose": choose},
        )
    )


def test_handler_entry_starts_an_independent_native_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _native_case(
        monkeypatch,
        "try:\n"
        "    raise LookupError\n"
        "except LookupError:\n"
        "    @capture\n"
        "    def sample(value: int):\n"
        "        yield value\n",
    )
    assert dis.Bytecode(case.module_code).exception_entries
    _assert_matches_native(case)


def test_replacing_decorator_does_not_change_raw_declaration_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = object()
    case = _native_case(
        monkeypatch,
        "@replace\n" "@capture\n" "async def sample(value: int):\n" "    yield value\n",
        namespace={"replace": lambda function: replacement},
    )
    assert case.namespace["sample"] is replacement
    assert case.function is not replacement
    _assert_matches_native(case)


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Deferred native annotation helper is Python 3.14+",
)
def test_helper_and_body_metadata_collision_does_not_require_name_heuristics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _native_case(
        monkeypatch,
        "def __annotate__(format: int, /) -> int:\n"
        "    return format\n"
        "capture(__annotate__)\n",
        target_name="__annotate__",
    )
    body = case.function.__code__
    (helper,) = tuple(
        value
        for value in case.module_code.co_consts
        if isinstance(value, CodeType) and value is not body
    )
    for attribute in (
        "co_name",
        "co_qualname",
        "co_argcount",
        "co_posonlyargcount",
        "co_varnames",
        "co_flags",
        "co_firstlineno",
    ):
        assert getattr(helper, attribute) == getattr(body, attribute)
    _assert_matches_native(case)


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Native annotate attachment backend is Python 3.14+",
)
@pytest.mark.parametrize(
    "declaration",
    (
        pytest.param(
            "def sample[T](value: T) -> T:\n    return value\n", id="ordinary"
        ),
        pytest.param(
            "async def sample[T](value: T) -> T:\n    return value\n", id="coroutine"
        ),
        pytest.param(
            "def sample[T: int = int](value: T = 7) -> T:\n    return value\n",
            id="bounds-and-defaults",
        ),
    ),
)
def test_annotated_generic_selects_body_not_type_parameter_wrapper(
    monkeypatch: pytest.MonkeyPatch, declaration: str
) -> None:
    _assert_matches_native(_native_case(monkeypatch, "@capture\n" + declaration))


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
def test_unannotated_generic_without_attachment_proof_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _native_case(
        monkeypatch,
        "@capture\ndef sample[T](value):\n    return value\n",
    )
    assert case.execution.mode is None
    assert case.execution.violation is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN


def test_duplicated_finally_creation_sites_remain_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _native_case(
        monkeypatch,
        "try:\n"
        "    side_effect()\n"
        "finally:\n"
        "    @capture\n"
        "    def sample(value: int) -> int:\n"
        "        return value\n",
        namespace={"side_effect": lambda: None},
    )
    # Native lowering emitted the same code at two distinct creation offsets.
    assert (
        sum(
            instruction.argval is case.function.__code__
            for instruction in dis.get_instructions(case.module_code)
        )
        == 2
    )
    assert case.execution.mode is None
    assert case.execution.violation is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN


@pytest.fixture
def native_region():
    if sys.version_info < (3, 14):
        pytest.skip("Native annotation attachment instructions require Python 3.14+")
    compilation = NativePythonCompilation(
        "def sample(value: int) -> int:\n    return value\n",
        "native_region.py",
    )
    code = compilation.compile()
    namespace = {}
    exec(code, namespace)
    body = namespace["sample"].__code__
    backend = NativeCreationBackend.current()
    assert isinstance(backend, CPython314CreationBackend)
    instructions = tuple(backend.instructions(code))
    index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.argval is body
    )
    load, make, attach = instructions[index : index + 3]
    assert make.opcode == dis.opmap["MAKE_FUNCTION"]
    assert attach.opcode == dis.opmap["SET_FUNCTION_ATTRIBUTE"]
    assert attach.arg == backend.annotate_flag
    return backend, compilation, code, load, make, attach


def _native_entry_label(kind: str) -> int:
    code = compile(
        "try:\n"
        "    if choice:\n        marker = 1\n"
        "    else:\n        marker = 2\n"
        "except Exception:\n    marker = 3\n",
        "native_entries.py",
        "exec",
    )
    bytecode = dis.Bytecode(code)
    instructions = tuple(bytecode)
    targets = {
        "jump": set(dis.findlabels(code.co_code)),
        "handler": {entry.target for entry in bytecode.exception_entries},
    }
    return next(
        instruction.label
        for instruction in instructions
        if instruction.offset in targets[kind]
    )


@pytest.mark.parametrize("entry_kind", ("jump", "handler"))
@pytest.mark.parametrize("stage", ("make", "attach"))
def test_incoming_native_entry_clears_predecessor_creation(
    native_region,
    entry_kind: str,
    stage: str,
) -> None:
    backend, compilation, _, load, make, attach = native_region
    emission = NativeCodeEmission(compilation.identity, load)
    current = backend.observe(None, load, emission)
    instruction = make
    if stage == "attach":
        current = backend.observe(current, make, None)
        instruction = attach
    # Use real native entry labels to impose an entry at this proof boundary.
    # This is a stop-obligation test, not a claim that the compiler emits it here.
    incoming = instruction._replace(label=_native_entry_label(entry_kind))
    assert incoming.is_jump_target
    assert backend.observe(current, incoming, None) is None
    assert emission.attachments == []
    assert not backend.proves_body(emission)


@pytest.mark.parametrize("opcode_name", ("STORE_NAME", "LOAD_CONST"))
def test_intervening_native_operation_ends_the_creation_region(
    native_region,
    opcode_name: str,
) -> None:
    backend, compilation, _, load, make, attach = native_region
    emission = NativeCodeEmission(compilation.identity, load)
    current = backend.observe(None, load, emission)
    current = backend.observe(current, make, None)
    intervening = next(
        instruction
        for instruction in dis.get_instructions(
            compile("marker = None", "intervening.py", "exec")
        )
        if instruction.opcode == dis.opmap[opcode_name]
    )
    assert not isinstance(intervening.argval, CodeType)
    current = backend.observe(current, intervening, None)
    assert current is None
    assert backend.observe(current, attach, None) is None
    assert not backend.proves_body(emission)


@pytest.mark.parametrize("entry_kind", ("jump", "handler"))
def test_code_load_at_native_entry_starts_a_new_region(
    native_region, entry_kind: str
) -> None:
    backend, compilation, _, load, make, attach = native_region
    old = NativeCodeEmission(compilation.identity, load)
    current = backend.observe(None, load, old)
    current = backend.observe(current, make, None)
    entry_load = load._replace(label=_native_entry_label(entry_kind))
    fresh = NativeCodeEmission(compilation.identity, entry_load)
    current = backend.observe(current, entry_load, fresh)
    assert current is fresh
    assert fresh.creation is None
    current = backend.observe(current, make, None)
    current = backend.observe(current, attach, None)
    assert current is fresh
    assert backend.proves_body(fresh)
    assert not backend.proves_body(old)


@pytest.mark.parametrize("operand", (0, 3, 1024))
def test_unadmitted_attribute_operand_does_not_preserve_target(
    native_region, operand: int
) -> None:
    backend, compilation, _, load, make, attach = native_region
    assert operand not in backend.attribute_flags
    emission = NativeCodeEmission(compilation.identity, load)
    current = backend.observe(None, load, emission)
    current = backend.observe(current, make, None)
    malformed = attach._replace(arg=operand, argval=operand)
    assert backend.observe(current, malformed, None) is None
    assert not backend.proves_body(emission)


def test_later_region_reset_does_not_erase_completed_attachment_witness(
    native_region,
) -> None:
    backend, compilation, code, load, make, attach = native_region
    emission = NativeCodeEmission(compilation.identity, load)
    current = backend.observe(None, load, emission)
    current = backend.observe(current, make, None)
    current = backend.observe(current, attach, None)
    store = next(
        instruction
        for instruction in backend.instructions(code)
        if instruction.opcode == dis.opmap["STORE_NAME"]
    )
    assert backend.observe(current, store, None) is None
    assert backend.proves_body(emission)


def test_unregistered_compiler_uses_span_only_without_annotation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(NativeCreationBackend, "__registry__", {})
    monkeypatch.delattr(dis, "FUNCTION_ATTR_FLAGS", raising=False)
    backend = NativeCreationBackend.current()
    assert isinstance(backend, SpanOnlyCreationBackend)
    source = "def sample():\n    return None\n"
    compilation = NativePythonCompilation(source, "unsupported_backend.py")
    span = SourceByteSpan.require_node(ast.parse(source).body[0])
    execution = compilation.execution_for(span)
    assert isinstance(execution, ExactNativeFunctionExecution)
    assert execution.violation is None


def test_projected_owner_retains_no_code_or_transient_emission_in_dict_values() -> None:
    source = "def sample(value: int) -> int:\n    return value\n"
    compilation = NativePythonCompilation(source, "compact_projection.py")
    compilation.execution_for(SourceByteSpan.require_node(ast.parse(source).body[0]))
    pending = [compilation]
    visited = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        assert not isinstance(
            value, (CodeType, dis.Instruction, NativeCodeEmission, ast.AST)
        )
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(
                getattr(value, declaration.name) for declaration in fields(value)
            )
            pending.extend(
                vars(value).values()
            )  # Includes lazy cached owner projections.
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
