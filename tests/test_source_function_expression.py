"""Function expressions use original creation evidence, never body execution."""

import ast
from dataclasses import replace
from types import FunctionType

import pytest

from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.source_execution import (
    SourceCompletionResolver,
    SourceFunctionCreationABC,
    SourceNativeOperandJoin,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.value_expression import (
    CompactValueExpression,
    FunctionExpression,
    LexicalValueReference,
)
from test_source_function_result import execution


def expression(environment):
    return next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Lambda)
    )


@pytest.mark.parametrize(
    "source",
    (
        "value = lambda: missing()\n",
        "value = lambda parameter=1: missing(parameter)\n",
        "value = lambda *, parameter=None: missing(parameter)\n",
        "class Owner:\n    value = lambda: missing()\n",
    ),
)
def test_original_function_expression_proves_creation_without_executing_body(source):
    environment = execution(source)
    node = expression(environment)
    result = environment.capture_value(node)
    result.require_closed()
    assert result.native_type is FunctionType
    assert environment.capture_value(node) is result
    # Creation evidence cannot justify a call or a destruction callback.
    with pytest.raises(ValueError):
        result.require_release()


def test_default_evaluation_is_required_even_when_the_body_is_deferred():
    environment = execution("value = lambda parameter=missing(): parameter\n")
    result = environment.capture_value(expression(environment))
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


def test_creation_in_an_uncalled_function_does_not_acquire_module_activation():
    environment = execution("def outer():\n    return lambda: None\n")
    result = environment.capture_value(expression(environment))
    assert isinstance(result, OpenCapturedReference)


def test_equal_but_nonoriginal_source_read_context_cannot_acquire_creation():
    environment = execution("value = lambda: None\n")
    node = expression(environment)
    read = environment.source.value_reads_by_node[node]
    with pytest.raises(ValueError):
        environment.kernel._read_use(read.use, replace(read.context), frozenset())


@pytest.mark.parametrize("warm", (False, True))
def test_equal_compiler_receipt_is_not_the_original_creation(monkeypatch, warm):
    environment = execution("value = lambda: None\n")
    node = expression(environment)
    compilation = environment.module.native_compilation
    native = compilation.execution_for(SourceByteSpan.require_node(node))
    if warm:
        environment.capture_value(node).require_closed()
    copied = replace(native)
    assert copied == native and copied is not native
    monkeypatch.setattr(type(compilation), "execution_for", lambda self, span: copied)
    with pytest.raises(ValueError):
        environment.capture_value(node).require_closed()


def test_native_operand_join_requires_original_inventory_and_creation():
    environment = execution("first = lambda: 1\nsecond = lambda: 2\n")
    nodes = [
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Lambda)
    ]
    stores = [
        SourceCompletionResolver(environment).resolve(binding)
        for binding in environment.entry.context.flow.mutations
    ]
    joins = [
        SourceNativeOperandJoin(store, environment.source.value_reads_by_node[node])
        for node, store in zip(nodes, stores, strict=True)
    ]
    values = [store.production.value for store in stores]
    first = joins[0].require_join(values[0])
    second = joins[1].require_join(values[1])
    assert isinstance(first, SourceFunctionCreationABC)
    assert isinstance(second, SourceFunctionCreationABC)
    assert first is not second
    assert first.native_execution is not second.native_execution
    for foreign in (values[1], replace(values[0])):
        with pytest.raises(ValueError):
            joins[0].require_join(foreign)


def test_syntax_projection_uses_nominal_inheritance_and_mro():
    class NameAlias(ast.Name):
        pass

    class FunctionAlias(ast.Lambda):
        pass

    class NameFirst(ast.Name, ast.Lambda):
        pass

    class FunctionFirst(ast.Lambda, ast.Name):
        pass

    for declaration in (NameAlias, NameFirst):
        node = declaration(id="value", ctx=ast.Load())
        assert CompactValueExpression.project(node) == LexicalValueReference("value")
    template = ast.parse("lambda: None", mode="eval").body
    for declaration in (FunctionAlias, FunctionFirst):
        node = declaration(args=template.args, body=template.body)
        assert isinstance(CompactValueExpression.project(node), FunctionExpression)


def test_module_builder_spelling_does_not_replace_the_actual_builtin():
    source = (
        "__build_class__ = lambda *args, **kwargs: None\n"
        "class Owner: pass\n"
        "Owner.changed = 1\n"
    )
    namespace = {}
    exec(source, namespace)
    assert type(namespace["Owner"]) is type
    assert namespace["Owner"].changed == 1
    environment = execution(source)
    environment.require_class_creation(environment.module.module.body[1])
