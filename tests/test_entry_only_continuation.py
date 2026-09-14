"""Event-free class bodies retain the original native boundary and return."""

from copy import copy
from dataclasses import replace
import dis
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.captured_reference import NamespaceMemberInventory
from nominal_refactor_advisor.native_compilation import (
    NativeContinuationWindow,
    NativePrimitiveOperation,
)
from nominal_refactor_advisor.source_execution import (
    EntryOnlyNamespaceContinuation,
    PreparedNamespaceContinuationABC,
    PreparedNamespaceTail,
)
from test_native_source_class_preparation import prepared_execution
from test_registry_native_registration_controls import native_control
from test_selected_store_continuation import body_entry


@pytest.mark.parametrize("body", ("pass", "pass\npass", "pass\n\n# comment\npass"))
def test_event_free_body_uses_original_prologue_boundary(body):
    entry = body_entry(body)
    tail = entry.native_tail
    assert type(tail) is EntryOnlyNamespaceContinuation
    assert entry.context.flow.mutations == ()
    start = entry.capture.prologue.require_body_start()
    assert tail.follows_source(start)
    assert not tail.follows_source(start - 1)
    assert tail.receipt.frame.is_body_of(entry.capture.body)
    assert tail.native_value(tail.receipt.value).require_native_scalar() is None
    assert tail.member("__qualname__").require_native_text() == "Family"
    assert tail.member("absent") is None
    assert tail.completed is None


def test_both_boundaries_share_all_namespace_interpretation():
    for name in (
        "member",
        "completed",
        "_native_initial_local",
        "bindings",
        "_require_native_production",
    ):
        assert name in vars(PreparedNamespaceContinuationABC)
        assert name not in vars(PreparedNamespaceTail)
        assert name not in vars(EntryOnlyNamespaceContinuation)


@pytest.mark.parametrize(
    "body", ("...", "object", "unknown()", "value = None", "if True:\n    pass")
)
def test_no_store_is_not_equivalent_to_no_source_work(body):
    entry = body_entry(body)
    with pytest.raises(ValueError):
        _ = EntryOnlyNamespaceContinuation(entry).completed


@pytest.mark.parametrize("warm", (False, True))
def test_original_suite_is_rechecked_after_queries(warm):
    entry = body_entry("pass\npass")
    tail = entry.native_tail
    if warm:
        tail.member("__qualname__").require_closed()
    entry.node.body.pop()
    with pytest.raises(ValueError, match="original body statements"):
        _ = tail.completed


def test_prologue_and_other_body_values_cannot_be_reinterpreted_after_entry():
    tail = body_entry("pass").native_tail
    other = body_entry("pass").native_tail
    for value in (
        *tail.entry.capture.prologue.values,
        other.receipt.value,
        copy(tail.receipt.value),
    ):
        with pytest.raises(ValueError, match="original production"):
            tail.native_value(value)
    for value in tail.receipt.values:
        if not tail.follows_source(value.instruction_offset):
            with pytest.raises(ValueError, match="precedes"):
                tail.native_value(value)


def test_native_body_query_authenticates_original_compilation_and_execution():
    entry = body_entry("pass")
    compilation = entry.execution.module.native_compilation
    body = entry.capture.body
    receipt = compilation.return_from(body)
    for candidate in (copy(body), body_entry("pass").capture.body):
        with pytest.raises(ValueError, match="canonical compilation receipt"):
            compilation.return_from(candidate)
    warmed = pickle.loads(pickle.dumps(compilation))
    original = warmed.execution_for(body.source_span)
    assert warmed.return_from(original).frame.is_body_of(original)
    assert warmed.return_from(original) == receipt
    with pytest.raises(ValueError):
        warmed.return_from(body)


def test_missing_ambiguous_or_late_return_cannot_supply_body_coverage():
    entry = body_entry("pass")
    tail = entry.native_tail
    compilation = entry.execution.module.native_compilation
    outcome = compilation.execution_outcome
    receipt = tail.receipt
    (scope,) = [scope for scope in outcome.scopes if scope.continuation is receipt]
    for returns in ((), (receipt, receipt), (replace(receipt, stores=()),)):
        compilation.__dict__["_execution_outcome"] = replace(
            outcome,
            scopes=tuple(replace(scope, continuation=value) for value in returns),
        )
        with pytest.raises(ValueError, match="return continuation|body boundary"):
            _ = tail.completed
    compilation.__dict__["_execution_outcome"] = outcome
    assert tail.completed is None


def test_prepared_native_metaclass_pass_body_does_not_admit_construction():
    env, (node,) = prepared_execution("class Family(metaclass=Creator):\n    pass\n")
    entry = env.class_entry(node)
    assert entry.native_tail.completed is None
    with pytest.raises(ValueError, match="construction over prepared inputs"):
        entry.result()


def test_nop_has_no_stack_or_storage_effect_and_jump_entry_still_breaks_tail():
    code = next(
        value
        for value in compile(
            "class Family:\n    pass\n    pass\n", "nop.py", "exec"
        ).co_consts
        if isinstance(value, CodeType)
    )
    nop = next(
        instruction
        for instruction in dis.get_instructions(code)
        if instruction.opname == "NOP"
    )
    window = NativeContinuationWindow()
    window.observe(nop, NativePrimitiveOperation.NOP)
    assert (
        window.operands.stack
        == window.operands.values
        == window.operands.bindings
        == []
    )
    loop = compile("while True:\n    pass\n", "jump.py", "exec")
    target = next(
        instruction
        for instruction in dis.get_instructions(loop)
        if instruction.is_jump_target
    )
    with pytest.raises(ValueError, match="uninterrupted"):
        window.observe(target, NativePrimitiveOperation[target.opname])


@pytest.mark.parametrize("body", ("pass", "pass\npass"))
def test_complete_namespace_matches_actual_class_frame_return(body):
    source = "class Family:\n" + "".join(
        "    " + line + "\n" for line in body.splitlines()
    )
    tail = body_entry(body).native_tail
    observed = native_control(
        "import json, sys\n"
        "observations = []\n"
        "def trace(frame, event, value):\n"
        "    if frame.f_code.co_name == 'Family' and event == 'return':\n"
        "        observations.append({'returned': value, 'types': {name: type(item).__name__ for name, item in frame.f_locals.items()}})\n"
        "    return trace\n"
        "sys.settrace(trace)\n"
        f"exec({source!r})\n"
        "sys.settrace(None)\n"
        "assert len(observations) == 1\n"
        "print(json.dumps(observations[0]))\n",
        False,
    )
    names = NamespaceMemberInventory(
        tail.execution.kernel, tail.entry, tail.entry.completion_prefix
    ).names | frozenset(binding.name for binding in tail.bindings)
    assert {name: tail.member(name).native_type.__name__ for name in names} == observed[
        "types"
    ]
    assert (
        tail.native_value(tail.receipt.value).require_native_scalar()
        is observed["returned"]
    )
