"""Native function stores and scalar stores share one original continuation."""

import ast
from copy import copy
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativePythonCompilation,
    NativeValueStoreStream,
    NativeStoreStream,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_native_scalar_store import store_for
from test_source_function_installation import module_function
from test_source_function_storage import prepared_function
from test_source_function_result import execution


def native_function(source):
    compilation = NativePythonCompilation(source, "function_return.py")
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    return compilation, compilation.execution_for(SourceByteSpan.require_node(node))


@pytest.mark.parametrize(
    "header",
    (
        "def chosen():",
        "def chosen(value=3, *, option=True):",
        "def chosen(value: int = 3) -> int:",
        "async def chosen():",
    ),
)
@pytest.mark.parametrize("in_class", (False, True))
def test_native_function_store_uses_its_original_enclosing_frame(header, in_class):
    source = header + "\n    raise RuntimeError('not executed')\n"
    if in_class:
        source = "class Family:\n" + "".join(
            "    " + line for line in source.splitlines(True)
        )
    compilation, function = native_function(source)
    result = compilation.return_after(function)
    assert result.frame is function.require_creation().frame
    assert result.continues(function.require_installation())
    assert result.value.require_native_scalar() is None
    assert compilation.return_after(function) is result


def test_scalar_and_function_bindings_share_the_declaration_owned_observer():
    assert NativeValueStoreStream.record_store is NativeStoreStream.record_store
    assert NativeValueStoreStream.return_receipt is NativeStoreStream.return_receipt
    source = "def chosen():\n    pass\nlater = 7\n"
    compilation, function = native_function(source)
    scalar = store_for(compilation, ast.parse(source).body[-1])
    result = compilation.return_after(function)
    assert result is scalar.require_return()
    assert result.continues(function.require_installation())
    assert result.continues(scalar.binding)


@pytest.mark.parametrize("tail", ("unknown + 1\n", "if condition:\n    unknown()\n"))
def test_unknown_suffix_does_not_gain_function_return_evidence(tail):
    compilation, function = native_function("def chosen():\n    pass\n" + tail)
    with pytest.raises(ValueError):
        compilation.return_after(function)


@pytest.mark.parametrize("decorator", ("", "@staticmethod\n"))
def test_observed_call_and_discard_do_not_admit_unknown_source_effects(decorator):
    env = execution(decorator + "def chosen(): pass\nunknown()\n")
    node = env.module.module.body[0]
    env.module.native_compilation.return_after_binding(
        SourceByteSpan.require_node(node), "chosen"
    )
    with pytest.raises(ValueError):
        env.required_prefix(env.source.module_context, None)


def test_decorator_result_is_not_the_raw_function_installation():
    compilation, function = native_function("@replace\ndef chosen():\n    pass\n")
    with pytest.raises(ValueError):
        compilation.return_after(function)


def test_foreign_and_copied_function_receipts_are_not_original_installations():
    source = "def chosen():\n    pass\n"
    compilation, function = native_function(source)
    _, foreign = native_function(source)
    for wrong in (foreign, replace(function)):
        with pytest.raises(ValueError, match="canonical compilation receipt"):
            compilation.return_after(wrong)


def test_copied_compilation_identity_and_ambiguous_returns_remain_open():
    compilation, function = native_function("def chosen():\n    pass\n")
    outcome = compilation._execution_outcome
    wrong_identity = replace(outcome, compilation=replace(compilation.identity))
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        wrong_identity.return_after(function)
    result = compilation.return_after(function)
    for duplicate in (result, replace(result)):
        ambiguous = replace(
            outcome,
            scopes=(
                replace(outcome.scopes[0], continuation=result),
                replace(outcome.scopes[0], continuation=duplicate),
            ),
        )
        with pytest.raises(ValueError, match="unique return continuation"):
            ambiguous.return_after(function)


def test_foreign_cached_outcome_cannot_replace_the_source_compilation_owner():
    source = "def chosen():\n    pass\n"
    compilation, _ = native_function(source)
    foreign = NativePythonCompilation(source, "foreign.py")
    function = foreign.execution_for(
        SourceByteSpan.require_node(ast.parse(source).body[0])
    )
    compilation.__dict__["_execution_outcome"] = foreign._execution_outcome
    for operation in (
        lambda: compilation.require_fresh_function_namespace(function),
        lambda: compilation.return_after(function),
        lambda: compilation.execution_for(function.source_span),
    ):
        with pytest.raises(ValueError, match="different compilation"):
            operation()


def test_warmed_serialisation_preserves_shared_scalar_and_function_receipt(monkeypatch):
    source = "def chosen():\n    pass\nlater = None\n"
    compilation, function = native_function(source)
    assert compilation.return_after(function).continues(function.require_installation())
    payload = pickle.dumps(compilation)

    def no_compile(self):
        raise AssertionError("Compact store continuations must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    node, assignment = ast.parse(source).body
    result = restored.return_after(
        restored.execution_for(SourceByteSpan.require_node(node))
    )
    assert result is store_for(restored, assignment).require_return()
    assert restored._execution_outcome == compilation._execution_outcome
    assert result == compilation.return_after(function)
    assert result is not compilation.return_after(function)
    assert not result.continues(function.require_installation())


def test_prepared_native_class_final_method_joins_complete_source_cut():
    env, entry, function = prepared_function(explicit_scope=False)
    result = function.return_continuation(entry.completion_prefix)
    assert result.frame is function.native_execution.require_creation().frame
    assert result.continues(
        function.require_native_installation(entry.completion_prefix)
    )
    assert result.value.require_native_scalar() is None
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()
    assert not env._pending


def test_function_continuation_rejects_incomplete_copied_and_foreign_source_cuts():
    env, function = module_function()
    other, _ = module_function()
    prefix = env.required_prefix(env.entry.context, None)
    for wrong in (
        function.parent_prefix,
        copy(prefix),
        other.required_prefix(other.entry.context, None),
    ):
        with pytest.raises(ValueError):
            function.return_continuation(wrong)


def test_completed_source_does_not_skip_effects_after_function_installation():
    env, function = module_function("def chosen():\n    pass\nunknown()\n")
    with pytest.raises(ValueError):
        function.return_continuation(env.required_prefix(env.entry.context, None))
