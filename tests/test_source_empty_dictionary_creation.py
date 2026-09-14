"""Fresh creation identity is not dictionary contents or item-store admission."""

import ast
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import CompactFlowValue
from nominal_refactor_advisor.source_execution import (
    SourceEmptyDictionaryCapture,
    SourceLiteralCapture,
    SourceModuleExecution,
)
from nominal_refactor_advisor.value_expression import (
    CompactValueExpression,
    EmptyDictionaryExpression,
    ValueExpressionResolverABC,
)


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("empty_creation.py"),
            "empty_creation",
            False,
            ast.parse(source),
            source,
        )
    )


class ConservativeResolver(ValueExpressionResolverABC[object, object]):
    def _lexical_value_resolution(self, reference, context):
        return context

    def _unproved_value_resolution(self, context):
        return context


@pytest.mark.parametrize("context", ("unknown", 7))
def test_shared_low_projection_retains_conservative_resolver_contract(context):
    value = CompactValueExpression.project(ast.parse("{}", mode="eval").body)
    assert type(value) is EmptyDictionaryExpression
    assert value.lexical_reference is None
    assert value.resolve_value(ConservativeResolver(), context) is context
    assert type(pickle.loads(pickle.dumps(value))) is EmptyDictionaryExpression
    with pytest.raises(ValueError):
        value.require_mapping_key()


def test_aliases_and_repeated_reads_share_the_actual_creation():
    environment = execution("original = {}\nalias = original\nother = {}\n")
    first, alias, other = environment.module.module.body
    original = environment.capture_value(first.value)
    captured_alias = environment.capture_value(alias.value)
    distinct = environment.capture_value(other.value)
    assert type(original) is SourceEmptyDictionaryCapture
    assert captured_alias is original
    assert environment.capture_value(first.value) is original
    assert original.proves_same_object(captured_alias)
    assert distinct is not original
    assert not original.proves_same_object(distinct)
    assert not isinstance(original, CapturedNativeObject)
    assert original.dictionary_namespace(environment.initial) is original
    assert original.as_builtin_namespace(environment.initial) is original
    assert original.captured_dictionary() is original
    first_binding = environment.entry.context.flow.mutations[0]
    assert (
        environment.kernel.assignment_value(environment.entry.context, first_binding)
        is original
    )


def test_trusted_native_alias_control():
    source = (
        "original = {}\nalias = original\nother = {}\n"
        "assert alias is original\nassert other is not original\n"
        "assert type(original) is dict\nassert not original\n"
    )
    native = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, check=False
    )
    assert native.returncode == 0, native.stderr


def test_empty_creation_does_not_allocate_analyser_literal_dict(monkeypatch):
    environment = execution("original = {}\n")

    def unexpected_literal_eval(node):
        pytest.fail("Empty creation must not derive target identity from literal_eval")

    monkeypatch.setattr(ast, "literal_eval", unexpected_literal_eval)
    result = environment.capture_value(environment.module.module.body[0].value)
    assert type(result) is SourceEmptyDictionaryCapture
    result.require_closed()


def test_creation_namespace_initial_absence_needs_actual_later_cut():
    environment = execution("original = {}\n")
    node = environment.module.module.body[0].value
    result = environment.capture_value(node)
    read = environment.source.value_reads_by_node[node]
    binding = environment.entry.context.flow.mutations[0]
    assert result.initial_names == frozenset()
    assert result.member("not_present") is None
    prefix = environment.required_prefix(read.context, binding.position)
    result.require_available(environment.kernel, prefix)
    at_creation = environment.required_prefix(read.context, read.use.position)
    with pytest.raises(ValueError, match="no unique occurrence"):
        result.require_available(environment.kernel, at_creation)


@pytest.mark.parametrize("defect", ("kernel", "initial", "owner"))
def test_foreign_execution_or_counterfeit_creation_is_not_admitted(defect):
    environment = execution("original = {}\n")
    other = execution("original = {}\n")
    node = environment.module.module.body[0].value
    result = environment.capture_value(node)
    other_result = other.capture_value(other.module.module.body[0].value)
    assert result is not other_result
    assert not result.proves_same_object(other_result)
    with pytest.raises(ValueError):
        if defect == "initial":
            result.require_admitted(other.initial)
        elif defect == "owner":
            replace(result).require_closed()
        else:
            result.require_available(
                other.kernel,
                other.required_prefix(
                    other.entry.context, other.entry.context.flow.mutations[0].position
                ),
            )


@pytest.mark.parametrize(
    "defect", ("copied_node", "copied_use", "duplicate_operation", "foreign_context")
)
def test_creation_requires_original_source_and_flow_relationship(defect):
    environment = execution("original = {}\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    if defect == "copied_node":
        copied = ast.parse("original = {}\n").body[0].value
        with pytest.raises(ValueError):
            environment.capture_value(copied).require_closed()
        return
    if defect == "copied_use":
        invalid = replace(read, use=replace(read.use))
    elif defect == "foreign_context":
        other = execution("original = {}\n")
        invalid = replace(read, context=other.entry.context)
    else:
        operation = environment.source_operation(read.context, read.use)
        environment.entry.__dict__["source"] = replace(
            environment.source, operations=(*environment.source.operations, operation)
        )
        invalid = read
    with pytest.raises(ValueError):
        environment.empty_dictionary_creation(invalid)


@pytest.mark.parametrize(
    "source",
    (
        "unknown()\noriginal = {}\n",
        "if flag:\n    original = {}\n",
        "for value in values:\n    original = {}\n",
        "def deferred():\n    original = {}\n",
    ),
)
def test_unknown_or_repeated_activation_remains_open(source):
    environment = execution(source)
    (node,) = tuple(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Dict)
    )
    result = environment.capture_value(node)
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


def test_unknown_compiler_does_not_gain_freshness_from_empty_shape(monkeypatch):
    environment = execution("original = {}\n")
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    result = environment.capture_value(environment.module.module.body[0].value)
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


@pytest.mark.parametrize("expression", ("{'a': 1}", "[]", "[1]", "None", "'text'"))
def test_other_literal_captures_remain_type_only(expression):
    environment = execution(f"original = {expression}\n")
    result = environment.capture_value(environment.module.module.body[0].value)
    assert type(result) is SourceLiteralCapture
    result.require_closed()
    with pytest.raises(ValueError):
        result.dictionary_namespace(environment.initial).require_closed()


def test_native_creation_law_rejects_nonempty_or_non_dictionary_source():
    backend = NativeCreationBackend.current()
    for source in ("{'a': 1}", "[]", "dict()"):
        with pytest.raises(ValueError):
            backend.require_empty_dictionary_creation(
                ast.parse(source, mode="eval").body
            )


def test_created_identity_does_not_admit_unproved_stored_value_release():
    environment = execution(
        "original = {}\noriginal['key'] = []\n"
        "original['key'] = 'replacement'\nalias = original\n"
    )
    first, _, _, alias = environment.module.module.body
    created = environment.capture_value(first.value)
    created.require_closed()
    result = environment.capture_value(alias.value)
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


def test_nonempty_value_read_cannot_request_empty_creation():
    environment = execution("original = {'key': 1}\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    with pytest.raises(ValueError):
        environment.empty_dictionary_creation(CompactFlowValue(read.context, read.use))


def test_wrong_low_projection_cannot_replace_native_syntax_evidence(monkeypatch):
    project = CompactValueExpression.project

    def wrongly_project_empty(expression):
        if isinstance(expression, ast.Dict):
            return EmptyDictionaryExpression()
        return project(expression)

    monkeypatch.setattr(
        CompactValueExpression, "project", staticmethod(wrongly_project_empty)
    )
    environment = execution("original = {'key': 1}\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    assert type(read.use.value) is EmptyDictionaryExpression
    owner = environment.empty_dictionary_creation(read)
    assert owner.operation.node is node
    with pytest.raises(ValueError, match="original empty syntax"):
        owner.require_closed()
    assert isinstance(environment.capture_value(node), OpenCapturedReference)


def test_admitted_plain_class_body_preserves_its_own_dictionary_creation():
    environment = execution("class Owner:\n    original = {}\n    alias = original\n")
    definition = environment.module.module.body[0]
    original, alias = definition.body
    result = environment.capture_value(original.value)
    assert type(result) is SourceEmptyDictionaryCapture
    assert environment.capture_value(alias.value) is result
    assert result.context is not environment.entry.context


def test_shared_value_operation_join_returns_the_original_receipt():
    environment = execution("value = 'text'\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    operation = environment.source.value_operation(read)
    assert operation is environment.source_operation(read.context, read.use)
    assert operation.node is node
    assert environment.source.value_operation(replace(read)) is operation


def test_shared_value_operation_does_not_accept_callable_reference_phase():
    environment = execution("import builtins\nvalue = builtins.property\n")
    node = environment.module.module.body[-1].value
    read = environment.source.reference_reads_by_node[node]
    assert environment.source_operation(read.context, read.use).node is node
    with pytest.raises(ValueError, match="original expression operation"):
        environment.source.value_operation(read)


def test_shared_value_operation_rejects_ambiguous_original_node():
    environment = execution("first = 'one'\nsecond = 'two'\n")
    first, second = environment.module.module.body
    read = environment.source.value_reads_by_node[first.value]
    second_read = environment.source.value_reads_by_node[second.value]
    second_operation = environment.source_operation(
        second_read.context, second_read.use
    )
    source = replace(
        environment.source,
        operations=tuple(
            (
                replace(operation, node=first.value)
                if operation is second_operation
                else operation
            )
            for operation in environment.source.operations
        ),
    )
    with pytest.raises(ValueError, match="original expression operation"):
        source.value_operation(read)
