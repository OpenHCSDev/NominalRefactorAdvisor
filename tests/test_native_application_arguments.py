"""Decorator inputs retain actual native predecessor identity and source order."""

from copy import copy
from dataclasses import replace
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.source_execution import SourceCreatedFunctionCapture
from test_native_function_creation import _case
from test_source_function_result import execution, function


@pytest.mark.parametrize("count", (1, 2, 4))
@pytest.mark.parametrize("in_class", (False, True))
def test_implicit_arguments_are_the_original_preceding_native_results(
    monkeypatch, count, in_class
):
    definition = "@wrap\n" * count + "def chosen(): raise AssertionError\n"
    source = (
        "class Owner:\n" + "".join("    " + s for s in definition.splitlines(True))
        if in_class
        else definition
    )
    compilation, receipt, code, inventory = _case(monkeypatch, source)
    (emission,) = (e for e in inventory.emissions if e.receipt is receipt)
    arguments, results = [], []

    def wrap(argument):
        arguments.append(argument)
        result = object()
        results.append(result)
        return result

    namespace = {"wrap": wrap}
    exec(code, namespace)  # The exact compiled authored fixture, never chosen().
    assert arguments[0].__code__ is emission.code
    assert all(arguments[i] is results[i - 1] for i in range(1, count))
    previous = receipt.require_creation()
    for application in receipt.require_applications():
        assert application.argument is previous
        assert application.frame is previous.frame
        assert application.instruction_offset > previous.instruction_offset
        previous = application
    installed = vars(namespace["Owner"])["chosen"] if in_class else namespace["chosen"]
    assert installed is results[-1]
    assert compilation.execution_for(receipt.source_span) is receipt


@pytest.mark.parametrize("damage", ("reorder", "duplicate", "rewind"))
def test_application_receipt_rejects_a_nonforward_native_chain(monkeypatch, damage):
    def corrupt(inventory):
        (emission,) = (e for e in inventory.emissions if e.applications)
        if damage == "reorder":
            emission.applications.reverse()
        elif damage == "duplicate":
            emission.applications.append(emission.applications[-1])
        else:
            item = emission.applications[0]
            emission.applications[0] = replace(
                item, call=item.call._replace(offset=emission.creation.offset)
            )

    with pytest.raises(ValueError, match="original implicit argument"):
        _case(monkeypatch, "@first\n@second\ndef chosen(): pass\n", damage=corrupt)


def test_pickled_application_chain_keeps_original_links_without_recompiling(
    monkeypatch,
):
    compilation, receipt, _, _ = _case(
        monkeypatch, "@first\n@second\ndef chosen(): pass\n"
    )
    payload = pickle.dumps(compilation)

    def no_compile(self):
        raise AssertionError("An original compact argument chain must not recompile")

    monkeypatch.setattr(NativePythonCompilation, "compile", no_compile)
    restored = pickle.loads(payload)
    actual = restored.execution_for(receipt.source_span)
    previous = actual.require_creation()
    for application in actual.require_applications():
        assert application.argument is previous
        previous = application
    with pytest.raises(ValueError, match="canonical compilation receipt"):
        restored.execution_outcome.require_function(copy(actual))


def test_long_native_argument_chain_reuses_iterative_value_comparison(monkeypatch):
    count = 1200
    compilation, receipt, _, _ = _case(
        monkeypatch, "@wrap\n" * count + "def chosen(): pass\n"
    )
    restored = pickle.loads(pickle.dumps(compilation))
    original = receipt.require_applications()[-1]
    other = restored.execution_for(receipt.source_span).require_applications()[-1]
    assert original is not other
    assert original == other
    assert hash(original) == hash(other)
    assert original != replace(other, argument=other.argument.argument)


def decorated():
    env = execution("@classmethod\n@staticmethod\ndef chosen(): pass\n")
    node, _, _ = function(env)
    result = SourceCreatedFunctionCapture(env, node).creation_results[-1]
    return env, node, result


def test_source_applications_join_each_original_native_input():
    _, _, result = decorated()
    result.require_closed()
    native = result.creation.native_execution
    applications = native.require_applications()
    assert result.native_value is applications[-1].operand_in(result.production)
    assert result.argument.native_value is applications[0].operand_in(result.production)
    assert applications[0].argument is native.require_creation()
    assert applications[1].argument is applications[0]


@pytest.mark.parametrize("warm", (False, True))
@pytest.mark.parametrize("damage", ("reverse", "duplicate", "omit"))
def test_source_application_order_is_revalidated_after_warming(warm, damage):
    _, node, result = decorated()
    if warm:
        result.require_closed()
    if damage == "reverse":
        node.decorator_list.reverse()
    elif damage == "duplicate":
        node.decorator_list[0] = node.decorator_list[1]
    else:
        node.decorator_list.pop()
    with pytest.raises(ValueError, match="original (source chain|decorator read)"):
        result.require_closed()


@pytest.mark.parametrize(
    "damage,message",
    (
        ("argument", "compiler predecessor chain"),
        ("frame", "different frame"),
    ),
)
def test_source_join_rejects_copied_native_argument_or_frame(damage, message):
    _, _, result = decorated()
    application = result.creation.native_execution.require_applications()[-1]
    original = getattr(application, damage)
    object.__setattr__(application, damage, copy(original))
    with pytest.raises(ValueError, match=message):
        result.require_closed()


def test_observed_implicit_argument_does_not_admit_unknown_decorator():
    env = execution("@unknown\ndef chosen(): pass\n")
    node, _, _ = function(env)
    creation = SourceCreatedFunctionCapture(env, node)
    application = creation.creation_results[-1]
    assert (
        creation.native_execution.require_applications()[-1].argument
        is creation.native_execution.require_creation()
    )
    with pytest.raises(ValueError):
        application.require_closed()
