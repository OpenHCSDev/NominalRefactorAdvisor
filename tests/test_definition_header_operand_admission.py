"""Actual header operands must complete before a definition result exists."""

import ast

import pytest

from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.product_flow import CompactValueUse
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_function_result import execution


def _source(scope, header, *, prefix="", body="pass"):
    function = f"{header}\n    {body}\n"
    if scope == "module":
        return prefix + function
    return (
        prefix
        + "class Holder:\n"
        + "".join("    " + line for line in function.splitlines(keepends=True))
    )


def _definition(environment):
    node = environment.module.module.body[-1]
    operation = environment.definition_operation(node)
    context = environment.context_for_owner(operation.owner)
    return environment.definition_result(context, operation.event)


def _runtime(source):
    namespace = {}
    exec(
        compile(source, "<authored-definition-header>", "exec", dont_inherit=True),
        namespace,
    )
    return namespace


def _function(namespace, scope):
    return namespace["chosen"] if scope == "module" else namespace["Holder"].chosen


@pytest.mark.parametrize("scope", ("module", "class"))
@pytest.mark.parametrize("keyword_only", (False, True))
def test_undefined_default_operand_cannot_produce_a_definition(scope, keyword_only):
    parameters = "*, value=missing" if keyword_only else "value=missing"
    source = _source(scope, f"def chosen({parameters}):")
    with pytest.raises(NameError, match="missing"):
        _runtime(source)
    environment = execution(source)
    with pytest.raises(ValueError, match="unproved"):
        _definition(environment).require_closed()


@pytest.mark.parametrize("scope", ("module", "class"))
@pytest.mark.parametrize("keyword_only", (False, True))
def test_bound_native_default_operand_is_captured_before_creation(scope, keyword_only):
    parameters = "*, value=known" if keyword_only else "value=known"
    source = _source(scope, f"def chosen({parameters}):", prefix="known = object\n")
    environment = execution(source)
    result = _definition(environment)
    result.require_closed()
    operation = environment.definition_operation(environment.module.module.body[-1])
    result.require_definition_identity(operation.event.target.owner)
    namespace = _runtime(source)
    function = _function(namespace, scope)
    default = (
        function.__kwdefaults__["value"] if keyword_only else function.__defaults__[0]
    )
    assert default is object


@pytest.mark.parametrize("prior_binding", (False, True))
def test_same_function_name_default_reads_the_prior_binding_only(prior_binding):
    source = "chosen = object\n" if prior_binding else ""
    source += "def chosen(value=chosen):\n    return value\n"
    environment = execution(source)
    if not prior_binding:
        with pytest.raises(NameError, match="chosen"):
            _runtime(source)
        with pytest.raises(ValueError, match="unproved"):
            _definition(environment).require_closed()
        return
    result = _definition(environment)
    result.require_closed()
    operation = environment.definition_operation(environment.module.module.body[-1])
    result.require_definition_identity(operation.event.target.owner)
    namespace = _runtime(source)
    assert namespace["chosen"].__defaults__ == (object,)
    assert namespace["chosen"]() is object
    assert namespace["chosen"] is not object


@pytest.mark.parametrize("scope", ("module", "class"))
@pytest.mark.parametrize("future", (False, True))
def test_annotation_operand_follows_actual_eager_future_or_lazy_phase(scope, future):
    source = _source(
        scope,
        "def chosen(value: missing) -> missing:",
        prefix="from __future__ import annotations\n" if future else "",
    )
    environment = execution(source)
    mode = ModuleAnnotationEvaluationMode.from_module(environment.module.module)
    if mode.annotations_execute_at_declaration:
        with pytest.raises(NameError, match="missing"):
            _runtime(source)
        with pytest.raises(ValueError, match="unproved"):
            _definition(environment).require_closed()
        return

    _definition(environment).require_closed()
    namespace = _runtime(source)
    function = _function(namespace, scope)
    if future:
        assert function.__annotations__ == {"value": "missing", "return": "missing"}
    else:
        # Lazy annotations are a later invocation, not a successful lookup at
        # the original definition cut and not an admitted callback execution.
        with pytest.raises(NameError, match="missing"):
            function.__annotations__


@pytest.mark.parametrize("scope", ("module", "class"))
def test_unknown_default_call_still_requires_its_original_execution(scope):
    source = _source(scope, "def chosen(value=missing()):")
    with pytest.raises(NameError, match="missing"):
        _runtime(source)
    environment = execution(source)
    with pytest.raises(ValueError, match="unproved"):
        _definition(environment).require_closed()


@pytest.mark.parametrize("scope", ("module", "class"))
def test_deferred_function_body_is_not_executed_as_a_header_operand(scope):
    source = _source(scope, "def chosen():", body="return missing()")
    environment = execution(source)
    _definition(environment).require_closed()
    namespace = _runtime(source)
    with pytest.raises(NameError, match="missing"):
        _function(namespace, scope)()


def _assert_input_receipts(environment, definition, expected_nodes):
    operation = environment.definition_operation(definition)
    context = environment.context_for_owner(operation.owner)
    target = operation.event.target
    assert len(target.input_uses) == len(expected_nodes)
    previous = None
    for use, node in zip(target.input_uses, expected_nodes, strict=True):
        captured = environment.source_operation(context, use)
        assert captured.node is node
        assert captured.event is use
        assert environment.source.value_reads_by_node[node].use is use
        assert environment.source.value_reads_by_node[node].context is context
        assert use.position.dominates(target.header_position)
        if previous is not None:
            assert previous.position.dominates(use.position)
        previous = use
    return target.input_uses


def test_sibling_inputs_exclude_prior_standalone_lambda_and_each_other():
    source = (
        "known = object\n"
        "standalone = lambda discarded=known: unexecuted_standalone()\n"
        "def first(value=known): pass\n"
        "def second(value=type): pass\n"
    )
    environment = execution(source)
    _, standalone, first, second = environment.module.module.body
    lambda_node = standalone.value
    assert isinstance(lambda_node, ast.Lambda)
    input_node = lambda_node.args.defaults[0]
    assert not any(
        isinstance(operation.event, CompactValueUse)
        for operation in environment.source.operations_by_node[input_node]
    )
    ordinary_read = environment.source.reference_reads_by_node[input_node]
    assert (
        environment.source_operation(ordinary_read.context, ordinary_read.use).node
        is input_node
    )
    first_inputs = _assert_input_receipts(environment, first, first.args.defaults)
    second_inputs = _assert_input_receipts(environment, second, second.args.defaults)
    assert ordinary_read.use.position.dominates(first_inputs[0].position)
    assert all(left is not right for left in first_inputs for right in second_inputs)
    assert lambda_node.body not in environment.source.operations_by_node
    assert lambda_node.body.func not in environment.source.reference_reads_by_node

    # This is source-receipt scoping, not admission of opaque lambda creation.
    namespace = _runtime(source)
    assert namespace["standalone"].__defaults__ == (object,)
    assert namespace["first"].__defaults__ == (object,)
    assert namespace["second"].__defaults__ == (type,)


def test_nested_lambda_default_receipts_follow_evaluation_without_entering_body():
    source = (
        "known = object\n"
        "def select(value): return value\n"
        "def chosen(callback=(lambda value=select(known): missing())): pass\n"
    )
    environment = execution(source)
    definition = environment.module.module.body[-1]
    lambda_node = definition.args.defaults[0]
    assert isinstance(lambda_node, ast.Lambda)
    (outer_input,) = _assert_input_receipts(environment, definition, (lambda_node,))
    inner_node = lambda_node.args.defaults[0]
    assert isinstance(inner_node, ast.Call)
    assert not any(
        isinstance(operation.event, CompactValueUse)
        for operation in environment.source.operations_by_node[inner_node]
    )
    ordinary_read = environment.source.reference_reads_by_node[inner_node.func]
    assert (
        environment.source_operation(ordinary_read.context, ordinary_read.use).node
        is inner_node.func
    )
    invocation = environment.source.call_operation(
        SourceByteSpan.require_node(inner_node)
    )
    assert invocation.node is inner_node
    assert (
        environment.source_operation(ordinary_read.context, invocation.event)
        is invocation
    )
    assert ordinary_read.use.position.dominates(invocation.position)
    assert invocation.position.dominates(outer_input.position)
    assert lambda_node.body not in environment.source.operations_by_node
    assert lambda_node.body.func not in environment.source.reference_reads_by_node
    namespace = _runtime(source)
    (callback,) = namespace["chosen"].__defaults__
    assert callback.__defaults__ == (object,)
    with pytest.raises(NameError, match="missing"):
        callback()


def test_class_bases_and_keywords_retain_actual_receipts_without_admitting_hooks():
    source = "class Base: pass\nclass Child(Base, metaclass=type): pass\n"
    environment = execution(source)
    base, child = environment.module.module.body
    _assert_input_receipts(environment, base, ())
    _assert_input_receipts(
        environment, child, (*child.bases, *(item.value for item in child.keywords))
    )
    with pytest.raises(
        ValueError, match="Captured object is not the required native declaration"
    ):
        _definition(environment).require_closed()
    namespace = _runtime(source)
    assert namespace["Child"].__bases__ == (namespace["Base"],)
    assert type(namespace["Child"]) is type
