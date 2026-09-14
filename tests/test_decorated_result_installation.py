"""Wrapped source results join native operands, storage and return separately."""

from copy import copy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.native_compilation import NativeCallMarker
from nominal_refactor_advisor.source_execution import (
    PreparedNamespaceTail,
    SourceCreatedFunctionCapture,
    SourceDefinitionCapture,
    SourceInstalledReturnABC,
    SourceCompletionResolver,
    SourceNativeExpressionABC,
)
from test_source_function_result import execution, function, native


def wrapped(decorators=("staticmethod",), in_class=False, parameters=""):
    definition = "".join(f"@{name}\n" for name in decorators)
    definition += (
        f"def chosen({parameters}):\n    raise RuntimeError('body must not run')\n"
    )
    source = (
        "class Owner:\n"
        + "".join("    " + line for line in definition.splitlines(True))
        if in_class
        else definition
    )
    env = execution(source)
    node, context, binding = function(env)
    raw = SourceCreatedFunctionCapture(env, node)
    return source, env, context, binding, raw.creation_results[-1]


@pytest.mark.parametrize("in_class", (False, True))
@pytest.mark.parametrize(
    "parameters",
    ("", "value=1", "value: object", "*, value=1", "*, left=1, right=None"),
)
@pytest.mark.parametrize("decorators", (("property",), ("classmethod", "staticmethod")))
def test_final_wrapped_result_supplies_its_original_store_and_return(
    in_class, parameters, decorators
):
    source, env, context, binding, result = wrapped(decorators, in_class, parameters)
    slot = "vars(Owner)['chosen']" if in_class else "chosen"
    native(source + f"assert type({slot}) is {decorators[0]}\n")
    prefix = env.required_prefix(context, None)
    installed = result.require_native_installation(prefix)
    assert installed is result.creation.native_execution.require_applied_installation()
    assert installed is result.production.binding
    assert installed.value is result.native_value
    assert isinstance(result, SourceInstalledReturnABC)
    assert not isinstance(result, SourceDefinitionCapture)
    assert result.return_continuation(prefix) is result.production.require_return()
    selected = SourceCompletionResolver(env).resolve(binding)
    assert selected.require_native_installation(prefix) is installed
    assert selected.return_continuation(prefix) is result.return_continuation(prefix)
    for operation in (
        result.source_definition,
        lambda: result.creation.require_native_installation(prefix),
        lambda: result.require_definition_identity(binding.target.owner),
    ):
        with pytest.raises(ValueError):
            operation()
    if in_class:
        owner = env.class_entry(env.module.module.body[0])
        tail = PreparedNamespaceTail(owner, result)
        assert tail.receipt is result.return_continuation(prefix)
        env.require_class_creation(owner.node)
    assert not env._pending


def test_intermediate_wrapper_does_not_borrow_the_final_wrapper_store():
    _, env, context, _, result = wrapped(("classmethod", "staticmethod"))
    with pytest.raises(ValueError, match="Only the final"):
        result.argument.require_native_installation(env.required_prefix(context, None))


def test_wrapped_installation_keeps_original_cut_and_frame_identity():
    _, env, context, _, result = wrapped()
    prefix = env.required_prefix(context, None)
    result.require_native_installation(prefix)
    for invalid in (copy(prefix), result.creation.parent_prefix):
        with pytest.raises(ValueError):
            result.require_native_installation(invalid)
    _, foreign, foreign_context, _, _ = wrapped()
    with pytest.raises(ValueError):
        result.require_native_installation(
            foreign.required_prefix(foreign_context, None)
        )


@pytest.mark.parametrize("damage", ("callee", "argument", "slot", "frame", "store"))
def test_warmed_original_receipts_reject_substituted_native_operands(damage):
    _, env, context, _, result = wrapped()
    prefix = env.required_prefix(context, None)
    result.require_native_installation(prefix)
    value = result.native_value
    if damage == "callee":
        object.__setattr__(value.callee, "name", "property")
    elif damage == "argument":
        copied = replace(value.arguments[0])
        object.__setattr__(value, "inputs", (value.callee, copied))
        object.__setattr__(value, "argument_slot", copied)
    elif damage == "slot":
        object.__setattr__(
            value,
            "argument_slot",
            NativeCallMarker(value.instruction_offset, value.source_span),
        )
    elif damage == "frame":
        object.__setattr__(result.production, "frame", copy(result.production.frame))
    else:
        object.__setattr__(
            result.production, "binding", replace(result.production.binding)
        )
    with pytest.raises(ValueError):
        result.require_native_installation(prefix)


def test_decorator_read_is_joined_at_its_original_read_not_default_completion(
    monkeypatch,
):
    source = "selected = staticmethod\n@selected\ndef chosen(value=1): pass\n"
    env = execution(source)
    node, context, binding = function(env)
    reads = []
    original = SourceNativeExpressionABC._native_initial_local

    def observe(join, name):
        reads.append(join.source_read)
        return original(join, name)

    monkeypatch.setattr(SourceNativeExpressionABC, "_native_initial_local", observe)
    result = env.definition_result(context, binding)
    result.require_native_installation(env.required_prefix(context, None))
    assert (
        result.native_value.callee.source_span.start_line
        == node.decorator_list[0].lineno
    )
    assert reads and all(read.use is result.operand for read in reads)
    assert result.operand.position != result.creation.parent_prefix.endpoint.position
