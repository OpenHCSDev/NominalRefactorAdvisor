"""Dictionary construction effects do not require literal class operands.

These checks concern original operand captures under a source entry premise, not
the identity of the resulting dictionary or general dictionary-store execution.
"""

import ast
import dis
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(expression, *, prefix=""):
    source = (
        "class Alpha:\n    pass\n"
        "class Beta:\n    pass\n" + prefix + f"REGISTRY = {expression}\n"
    )
    module = ParsedModule(
        Path("/repo/dictionary_construction.py"),
        "dictionary_construction",
        False,
        ast.parse(source),
        source,
    )
    execution = SourceModuleExecution.from_module(module)
    assignment = module.module.body[-1]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Dict)
    return execution, assignment.value


def _require_original_class(execution, node):
    assert isinstance(node, ast.Name)
    declaration = next(
        item
        for item in execution.module.module.body
        if isinstance(item, ast.ClassDef) and item.name == node.id
    )
    operation = execution.definition_operation(declaration)
    captured = execution.capture(node)
    captured.require_closed()
    captured.require_definition_identity(operation.event.target.owner)
    context, binding = captured.source_definition()
    assert binding is operation.event
    assert context is execution.context_for_owner(operation.owner)
    return captured


def _require_dictionary_effects(execution, dictionary):
    evaluation = next(
        item for item in execution.source.evaluations if item.node is dictionary
    )
    context = execution.context_for_owner(evaluation.owner)
    execution.required_prefix(context, evaluation.exit)


@pytest.mark.parametrize(
    "keys",
    (
        ("'alpha'", "'beta'"),
        ("1", "2"),
        ("None", "..."),
        ("b'alpha'", "b'beta'"),
        ("(1, 'alpha')", "(2, 'beta')"),
        ("1+2j", "3+4j"),
    ),
)
def test_distinct_literal_keys_allow_original_class_operand_capture(keys):
    execution, dictionary = _execution(f"{{{keys[0]}: Alpha, {keys[1]}: Beta}}")
    for node in dictionary.values:
        _require_original_class(execution, node)


@pytest.mark.parametrize("size", (1, 2, 20))
def test_compiler_chunking_does_not_require_values_to_be_literals(size):
    expression = (
        "{"
        + ", ".join(
            f"'key{index}': {'Alpha' if index % 2 == 0 else 'Beta'}"
            for index in range(size)
        )
        + "}"
    )
    execution, dictionary = _execution(expression)
    captures = tuple(
        _require_original_class(execution, node) for node in dictionary.values
    )
    assert len(captures) == size
    # The outer construction claim must not fabricate a target-runtime dictionary.
    with pytest.raises(ValueError):
        execution.capture_value(dictionary).require_definition_identity(
            execution.definition_operation(
                execution.module.module.body[0]
            ).event.target.owner
        )


@pytest.mark.parametrize(
    "left,right",
    (
        ("'same'", "'same'"),
        ("1", "True"),
        ("1", "1.0"),
        ("0", "-0.0"),
        ("(1, (2,))", "(True, (2.0,))"),
    ),
)
def test_equal_literal_keys_do_not_assume_overwritten_value_release(left, right):
    execution, dictionary = _execution(f"{{{left}: Alpha, {right}: Beta}}")
    # The first operand is read before dictionary assembly. This does not prove
    # that releasing a subsequently displaced entry is inert.
    _require_original_class(execution, dictionary.values[0])
    # Interior observer cuts remain conservative without an exact native
    # assembly schedule; post-expression closure must retain the release gate.
    with pytest.raises(ValueError):
        _require_original_class(execution, dictionary.values[1])
    with pytest.raises(ValueError):
        _require_dictionary_effects(execution, dictionary)
    instructions = tuple(
        dis.get_instructions(
            compile(
                f"REGISTRY = {{{left}: Alpha, {right}: Beta}}",
                "<native-dictionary-order>",
                "exec",
                dont_inherit=True,
            )
        )
    )
    first_assembly = next(
        index
        for index, item in enumerate(instructions)
        if item.opname in ("BUILD_MAP", "BUILD_CONST_KEY_MAP")
    )
    assert {
        item.argval
        for item in instructions[:first_assembly]
        if item.opname == "LOAD_NAME"
    } == {"Alpha", "Beta"}


def test_duplicate_literal_values_allow_later_original_class_capture():
    execution, dictionary = _execution(
        "{'same': 'old', 'same': None, 'same': 'new', 'class': Alpha}"
    )
    _require_original_class(execution, dictionary.values[3])


def test_duplicate_class_value_replaced_by_literal_still_requires_release_proof():
    execution, dictionary = _execution("{'same': Alpha, 'same': 1, 'class': Beta}")
    with pytest.raises(ValueError):
        _require_original_class(execution, dictionary.values[2])


@pytest.mark.parametrize(
    "expression,prefix,first_operand_precedes_construction",
    (
        ("{key: Alpha, 'beta': Beta}", "key = 'alpha'\n", False),
        (
            "{make_key(): Alpha, 'beta': Beta}",
            "def make_key():\n    return 'alpha'\n",
            False,
        ),
        ("{**other, 'beta': Beta}", "other = {}\n", False),
        ("{'alpha': Alpha, **other}", "other = {}\n", True),
    ),
)
def test_unknown_keys_and_unpacking_remain_unproved(
    expression, prefix, first_operand_precedes_construction
):
    execution, dictionary = _execution(expression, prefix=prefix)
    original_class_values = tuple(
        node
        for node in dictionary.values
        if isinstance(node, ast.Name) and node.id in ("Alpha", "Beta")
    )
    assert original_class_values
    if first_operand_precedes_construction:
        _require_original_class(execution, original_class_values[0])
        original_class_values = original_class_values[1:]
    for node in original_class_values:
        with pytest.raises(ValueError):
            _require_original_class(execution, node)
    with pytest.raises(ValueError):
        _require_dictionary_effects(execution, dictionary)


def test_distinct_keys_do_not_hide_a_preceding_effectful_value():
    execution, dictionary = _execution(
        "{'alpha': effect(), 'beta': Beta}",
        prefix=(
            "def effect():\n"
            "    global changed\n"
            "    changed = True\n"
            "    return Alpha\n"
        ),
    )
    # Demand the real later operand, whose strict prefix includes effect().
    with pytest.raises(ValueError):
        _require_original_class(execution, dictionary.values[1])


def test_foreign_equal_span_operand_does_not_gain_source_identity():
    execution, dictionary = _execution("{'alpha': Alpha, 'beta': Beta}")
    foreign_module = ast.parse(execution.module.source)
    foreign_dictionary = foreign_module.body[-1].value
    original = dictionary.values[0]
    foreign = foreign_dictionary.values[0]
    assert ast.dump(original, include_attributes=True) == ast.dump(
        foreign, include_attributes=True
    )
    _require_original_class(execution, original)
    with pytest.raises(ValueError):
        _require_original_class(execution, foreign)


def test_unsupported_backend_rejects_dictionary_construction_specifically(monkeypatch):
    execution, dictionary = _execution("{'alpha': Alpha, 'beta': Beta}")
    # Establish class receipts first, so missing dictionary capability is the
    # tested boundary rather than unsupported class compilation or activation.
    for node in execution.module.module.body:
        if isinstance(node, ast.ClassDef):
            execution.require_class_creation(node)
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    # The native class identity is already established before dictionary entry.
    _require_original_class(execution, dictionary.values[0])
    with pytest.raises(ValueError) as failure:
        _require_dictionary_effects(execution, dictionary)
    causes = []
    error = failure.value
    while error is not None:
        causes.append(str(error))
        error = error.__cause__
    assert "Native dictionary construction remains unproved" in causes


@pytest.mark.parametrize("size", (2, 20))
def test_native_distinct_literal_key_construction_preserves_value_order(size):
    # Only run this small authored fixture in the subprocess, never analyzed files.
    source = (
        "events = []\n"
        "class Alpha: pass\n"
        "class Beta: pass\n"
        "def value(index):\n"
        "    events.append(index)\n"
        "    return Alpha if index % 2 == 0 else Beta\n"
        "result = {"
        + ", ".join(f"'key{index}': value({index})" for index in range(size))
        + "}\n"
        + f"assert events == list(range({size}))\n"
        + f"assert tuple(result) == tuple('key' + str(i) for i in range({size}))\n"
        + "assert all(result[key] is (Alpha if i % 2 == 0 else Beta) "
        "for i, key in enumerate(result))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
