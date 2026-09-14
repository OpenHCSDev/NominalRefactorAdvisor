"""Final source boundaries retain original statement order and activation."""

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_native_source_class_preparation import SOURCE, prepared_execution


def entry_for(body):
    module = SourceModule(
        path=Path("completion.py"),
        module_name="completion",
        source="class Family:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n",
    ).parse()
    environment = SourceModuleExecution.from_module(module)
    return environment.class_entry(module.module.body[0])


@pytest.mark.parametrize(
    "body",
    (
        "pass",
        "key = None",
        "enabled = True",
        "label = 'value'",
        "def method(self):\n    raise RuntimeError('never execute target body')",
        "async def method(self):\n    pass",
        "key = None\npass",
        "__registry_key__ = 'key'\n__skip_if_no_key__ = True\nkey = None",
    ),
)
def test_final_evaluation_is_original_for_every_supported_statement_family(body):
    entry = entry_for(body)
    actual = entry.final_evaluation
    assert actual.node is entry.node.body[-1]
    assert actual.owner is entry.context.flow.owner
    assert any(actual is original for original in entry.execution.source.evaluations)
    assert entry.final_evaluation is actual
    assert entry.completion_prefix is entry.execution.required_prefix(
        entry.context, None
    )
    assert not entry.execution._pending


def test_prepared_native_root_completion_does_not_claim_construction():
    environment, (root, *_) = prepared_execution(SOURCE, conditions=True)
    entry = environment.class_entry(root)
    assert entry.final_evaluation.node is root.body[-1]
    with pytest.raises(
        ValueError, match="External source interference remains unproved"
    ):
        entry.result()


@pytest.mark.parametrize("body", ("key = None\npass", "key = None\nlater = 7"))
@pytest.mark.parametrize("edit", ("drop", "reverse", "copy", "append", "duplicate"))
@pytest.mark.parametrize("warm", (False, True))
def test_changed_statement_list_cannot_borrow_original_completion(body, edit, warm):
    entry = entry_for(body)
    if warm:
        _ = entry.final_evaluation
    if edit == "drop":
        entry.node.body.pop()
    elif edit == "reverse":
        entry.node.body.reverse()
    elif edit == "copy":
        entry.node.body[-1] = deepcopy(entry.node.body[-1])
    elif edit == "append":
        entry.node.body.append(ast.Pass())
    else:
        entry.node.body.append(entry.node.body[-1])
    with pytest.raises(ValueError, match="original body statements"):
        _ = entry.final_evaluation


def test_other_activation_keeps_its_own_completion():
    first = entry_for("key = None")
    second = entry_for("key = None")
    assert first.final_evaluation is not second.final_evaluation
    assert first.final_evaluation.owner is not second.final_evaluation.owner
    assert (
        first.completion_prefix.endpoint.frame
        is not second.completion_prefix.endpoint.frame
    )


def test_body_with_unproved_effect_does_not_acquire_completed_cut():
    entry = entry_for("unknown()\nkey = None")
    with pytest.raises(ValueError):
        _ = entry.final_evaluation
