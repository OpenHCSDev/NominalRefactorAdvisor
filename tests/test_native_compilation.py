"""Native compiler provenance for raw function code, without module execution."""

import ast
import builtins
from dataclasses import fields, is_dataclass
import inspect
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_semantics import CodemodBackend
from nominal_refactor_advisor.native_compilation import (
    ExactNativeFunctionExecution,
    NativeCreationBackend,
    NativeExecutionUnavailable,
    NativeFunctionExecutionMode,
    NativePythonCompilation,
    OpenNativeFunctionExecution,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.source_identity import python_source_cache_signature


def _module(source: str) -> ParsedModule:
    return ParsedModule(
        path=Path("native_fixture.py"),
        module_name="native_fixture",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def _definitions(module: ParsedModule) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(module.module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


@pytest.mark.parametrize(
    "source, mode, flag",
    (
        ("def sample():\n    return None\n", NativeFunctionExecutionMode.ORDINARY, 0),
        (
            "async def sample():\n    return None\n",
            NativeFunctionExecutionMode.COROUTINE,
            inspect.CO_COROUTINE,
        ),
        (
            "def sample():\n    yield 1\n",
            NativeFunctionExecutionMode.GENERATOR,
            inspect.CO_GENERATOR,
        ),
        (
            "async def sample():\n    yield 1\n",
            NativeFunctionExecutionMode.ASYNC_GENERATOR,
            inspect.CO_ASYNC_GENERATOR,
        ),
        (
            "def sample():\n    if False:\n        yield 1\n    return None\n",
            NativeFunctionExecutionMode.GENERATOR,
            inspect.CO_GENERATOR,
        ),
    ),
)
def test_mode_comes_from_native_raw_code_flags(
    source: str, mode: NativeFunctionExecutionMode, flag: int
) -> None:
    module = _module(source)
    (definition,) = _definitions(module)
    span = SourceByteSpan.require_node(definition)
    execution = module.native_compilation.execution_for(span)
    assert isinstance(execution, ExactNativeFunctionExecution)
    assert execution.mode is mode
    assert execution.source_span == span
    assert module.native_compilation.execution_for(span) is execution
    assert execution.compilation is module.native_compilation.identity
    assert execution.violation is None
    if flag:
        assert execution.native_flags & flag
    namespace = {}
    exec(source, namespace)
    assert execution.native_flags == namespace["sample"].__code__.co_flags


def test_lazy_compilation_is_shared_without_executing_module_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        "import nonexistent_nra_probe_dependency\n"
        "raise RuntimeError('must not run')\n"
        "def sample():\n    return None\n"
    )
    span = SourceByteSpan.require_node(_definitions(module)[0])
    native_compile = builtins.compile
    calls = []

    def observed_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return native_compile(*args, **kwargs)

    monkeypatch.setattr(builtins, "compile", observed_compile)
    compilation = module.native_compilation
    assert compilation is module.native_compilation
    assert not calls
    compilation.execution_for(span)
    compilation.execution_for(span)
    assert len(calls) == 1
    assert calls[0][1] == {"dont_inherit": True, "optimize": 0}


def test_repeated_decorated_definitions_use_full_spans_not_names_or_first_lines() -> (
    None
):
    module = _module(
        "if choose:\n"
        "    @decorate\n"
        "    def sample():\n"
        "        yield 1\n"
        "else:\n"
        "    @factory(lambda: None)\n"
        "    async def sample():\n"
        "        return None\n"
    )
    first, second = _definitions(module)
    compilation = module.native_compilation
    assert compilation.execution_for(SourceByteSpan.require_node(first)).mode is (
        NativeFunctionExecutionMode.GENERATOR
    )
    assert compilation.execution_for(SourceByteSpan.require_node(second)).mode is (
        NativeFunctionExecutionMode.COROUTINE
    )
    assert (
        compilation.execution_for(
            SourceByteSpan.require_node(first.decorator_list[0])
        ).mode
        is None
    )


def test_nested_default_yield_changes_outer_not_inner_execution() -> None:
    module = _module(
        "def outer():\n"
        "    value = 1\n"
        "    def inner(argument=(yield 1)):\n"
        "        nonlocal value\n"
        "        return value\n"
        "    return inner\n"
    )
    outer, inner = _definitions(module)
    assert module.native_compilation.execution_for(
        SourceByteSpan.require_node(outer)
    ).mode is (NativeFunctionExecutionMode.GENERATOR)
    assert module.native_compilation.execution_for(
        SourceByteSpan.require_node(inner)
    ).mode is (NativeFunctionExecutionMode.ORDINARY)


def test_nested_function_and_lambda_yields_do_not_change_outer_execution() -> None:
    module = _module(
        "def outer():\n"
        "    deferred = lambda: (yield 1)\n"
        "    def inner():\n"
        "        yield 1\n"
        "    return deferred\n"
    )
    outer, inner = _definitions(module)
    assert module.native_compilation.execution_for(
        SourceByteSpan.require_node(outer)
    ).mode is (NativeFunctionExecutionMode.ORDINARY)
    assert module.native_compilation.execution_for(
        SourceByteSpan.require_node(inner)
    ).mode is (NativeFunctionExecutionMode.GENERATOR)


def test_eliminated_definition_does_not_recover_by_name_or_line() -> None:
    module = _module("if False:\n    def sample():\n        yield 1\n")
    result = module.native_compilation.execution_for(
        SourceByteSpan.require_node(_definitions(module)[0])
    )
    assert isinstance(result, OpenNativeFunctionExecution)
    assert result.mode is None
    assert result.violation is NativeExecutionUnavailable.NO_EMITTED_CODE


def test_leaf_bodies_are_not_disassembled_to_find_nonexistent_child_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        "def ordinary():\n    return None\n"
        "def generator():\n    yield 1\n"
        "async def coroutine():\n    return None\n"
        "async def async_generator():\n    yield 1\n"
    )
    backend = NativeCreationBackend.current()
    native_instructions = backend.instructions
    disassembled = []

    def observed_instructions(self, code):
        assert any(isinstance(value, CodeType) for value in code.co_consts)
        disassembled.append(code.co_name)
        return native_instructions(code)

    monkeypatch.setattr(type(backend), "instructions", observed_instructions)
    modes = tuple(
        module.native_compilation.execution_for(SourceByteSpan.require_node(node)).mode
        for node in _definitions(module)
    )
    assert modes == (
        NativeFunctionExecutionMode.ORDINARY,
        NativeFunctionExecutionMode.GENERATOR,
        NativeFunctionExecutionMode.COROUTINE,
        NativeFunctionExecutionMode.ASYNC_GENERATOR,
    )
    assert disassembled == ["<module>"]


def test_child_constants_still_require_emitted_instruction_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        "def outer():\n"
        "    if False:\n"
        "        def absent():\n"
        "            yield 1\n"
        "    def inner():\n"
        "        yield 1\n"
        "    return inner\n"
    )
    backend = NativeCreationBackend.current()
    native_instructions = backend.instructions
    disassembled = []

    def observed_instructions(self, code):
        assert any(isinstance(value, CodeType) for value in code.co_consts)
        disassembled.append(code.co_name)
        return native_instructions(code)

    monkeypatch.setattr(type(backend), "instructions", observed_instructions)
    definitions = {node.name: node for node in _definitions(module)}
    compilation = module.native_compilation
    absent = compilation.execution_for(
        SourceByteSpan.require_node(definitions["absent"])
    )
    assert absent.violation is NativeExecutionUnavailable.NO_EMITTED_CODE
    assert (
        compilation.execution_for(
            SourceByteSpan.require_node(definitions["inner"])
        ).mode
        is NativeFunctionExecutionMode.GENERATOR
    )
    assert disassembled == ["<module>", "outer"]


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Native type parameters require Python 3.12+"
)
def test_generated_generic_scopes_do_not_overwrite_ambiguous_source_evidence() -> None:
    module = _module("async def sample[T](value):\n    return value\n")
    result = module.native_compilation.execution_for(
        SourceByteSpan.require_node(_definitions(module)[0])
    )
    assert isinstance(result, OpenNativeFunctionExecution)
    assert result.mode is None
    assert result.violation is NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN


def test_compiler_rejection_is_explicit_and_validation_preserves_syntax_error() -> None:
    source = "def sample():\n    nonlocal absent\n    return absent\n"
    module = _module(source)
    result = module.native_compilation.execution_for(
        SourceByteSpan.require_node(_definitions(module)[0])
    )
    assert result.violation is NativeExecutionUnavailable.COMPILATION_REJECTED
    assert result.mode is None
    with pytest.raises(SyntaxError, match="nonlocal"):
        module.native_compilation.compile()
    with pytest.raises(SyntaxError, match="nonlocal"):
        CodemodBackend.AST_SPAN.validate_source(source, module.file_path)


def test_no_debug_ranges_preserves_unavailable_geometry() -> None:
    command = """
import ast
from nominal_refactor_advisor.native_compilation import NativePythonCompilation, NativeExecutionUnavailable
from nominal_refactor_advisor.source_geometry import SourceByteSpan
source = "def sample():\\n    return None\\n"
span = SourceByteSpan.require_node(ast.parse(source).body[0])
result = NativePythonCompilation(source, "no_ranges.py").execution_for(span)
assert result.mode is None
assert result.violation is NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES
"""
    subprocess.run([sys.executable, "-X", "no_debug_ranges", "-c", command], check=True)


def _assert_compact(value: object) -> None:
    assert not isinstance(value, (ast.AST, CodeType))
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_compact(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_compact(item)


def test_serialized_execution_keeps_source_provenance_without_ast_or_code() -> None:
    module = _module("def sample():\n    return None\n")
    result = module.native_compilation.execution_for(
        SourceByteSpan.require_node(_definitions(module)[0])
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    _assert_compact(restored)
    changed = NativePythonCompilation(
        module.source + "# different source\n", module.file_path
    )
    assert changed.identity != result.compilation
    assert result.compilation.interpreter == (sys.implementation.name, sys.version)
    assert result.compilation.source_digest == python_source_cache_signature(
        module.source
    )


def test_exact_source_hash_reexport_is_the_same_lower_authority() -> None:
    from nominal_refactor_advisor.ast_tools import (
        python_source_cache_signature as public_signature,
    )

    assert public_signature is python_source_cache_signature


def test_compilation_reuses_canonical_source_path_policy() -> None:
    compilation = NativePythonCompilation("", "package\\module.py")
    assert compilation.file_path == "package/module.py"
    assert compilation.identity.file_path == compilation.file_path


def _queried_module_mode(module: ParsedModule) -> NativeFunctionExecutionMode | None:
    span = SourceByteSpan.require_node(_definitions(module)[0])
    return module.native_compilation.execution_for(span).mode


@pytest.mark.parametrize(
    "source, expected",
    (
        (
            "async def sample():\n    return None\n",
            NativeFunctionExecutionMode.COROUTINE,
        ),
        ("def sample():\n    nonlocal absent\n", None),
    ),
)
def test_queried_module_survives_pickle_and_spawn_transfer(
    source: str, expected: NativeFunctionExecutionMode | None
) -> None:
    module = _module(source)
    assert _queried_module_mode(module) is expected
    restored = pickle.loads(pickle.dumps(module))
    assert _queried_module_mode(restored) is expected
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_queried_module_mode, (module,)) is expected
