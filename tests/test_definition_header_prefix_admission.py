"""Consumed definition headers are obligations even without reading the result."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.class_namespace import DefinitionHeaderEffect
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from test_source_definition_result import execution


def runtime(source):
    namespace = {}
    exec(
        compile(source, "<header-prefix-fixture>", "exec", dont_inherit=True), namespace
    )
    return namespace


def require_endpoint(environment, endpoint):
    if endpoint == "prefix":
        result = environment.admit(environment.source.module_context, None)
        if isinstance(result, OpenCapturedReference):
            result.require_closed()
        return result
    node = environment.module.module.body[-1]
    assert isinstance(node, ast.Assign)
    result = environment.capture(node.value)
    result.require_native_identity(NativeDeclaration(object))
    return result


@pytest.mark.parametrize("endpoint", ("prefix", "later_read"))
@pytest.mark.parametrize("parameters", ("value=absent", "*, value=absent"))
@pytest.mark.parametrize("asynchronous", (False, True))
def test_missing_default_prevents_prefix_and_unrelated_later_read(
    endpoint, parameters, asynchronous
):
    header = "async def" if asynchronous else "def"
    text = f"{header} unused({parameters}): pass\nlater = object\n"
    with pytest.raises(NameError, match="absent"):
        runtime(text)
    environment = execution(text)
    with pytest.raises(ValueError, match="unproved"):
        require_endpoint(environment, endpoint)
    assert not environment._pending


@pytest.mark.parametrize("endpoint", ("prefix", "later_read"))
@pytest.mark.parametrize("future", (False, True))
def test_annotation_prefix_uses_actual_native_annotation_phase(endpoint, future):
    text = "from __future__ import annotations\n" if future else ""
    text += "def unused(value: absent) -> absent: pass\nlater = object\n"
    environment = execution(text)
    mode = ModuleAnnotationEvaluationMode.from_module(environment.module.module)
    if mode.annotations_execute_at_declaration:
        with pytest.raises(NameError, match="absent"):
            runtime(text)
        with pytest.raises(ValueError, match="unproved"):
            require_endpoint(environment, endpoint)
    else:
        namespace = runtime(text)
        assert namespace["later"] is object
        require_endpoint(environment, endpoint)
        if future:
            assert namespace["unused"].__annotations__ == {
                "value": "absent",
                "return": "absent",
            }
        else:
            with pytest.raises(NameError, match="absent"):
                _ = namespace["unused"].__annotations__
    assert not environment._pending


@pytest.mark.parametrize("endpoint", ("prefix", "later_read"))
@pytest.mark.parametrize("default", ("object", "absent()"))
def test_native_default_and_unknown_call_keep_distinct_prefix_obligations(
    endpoint, default
):
    text = f"def unused(value={default}): return never_called\nlater = object\n"
    environment = execution(text)
    if default == "object":
        namespace = runtime(text)
        assert namespace["unused"].__defaults__ == (object,)
        require_endpoint(environment, endpoint)
        with pytest.raises(NameError, match="never_called"):
            namespace["unused"]()
    else:
        with pytest.raises(NameError, match="absent"):
            runtime(text)
        with pytest.raises(ValueError, match="unproved"):
            require_endpoint(environment, endpoint)
    assert not environment._pending


@pytest.mark.parametrize("keyword_only", (False, True))
def test_later_header_operand_cannot_bypass_an_earlier_missing_default(keyword_only):
    parameters = "first=absent, second=object"
    if keyword_only:
        parameters = "*, " + parameters
    text = f"def unused({parameters}): pass\n"
    with pytest.raises(NameError, match="absent"):
        runtime(text)
    environment = execution(text)
    function = environment.module.module.body[0]
    defaults = function.args.kw_defaults if keyword_only else function.args.defaults
    with pytest.raises(ValueError, match="unproved"):
        environment.capture(defaults[-1]).require_native_identity(
            NativeDeclaration(object)
        )
    assert not environment._pending


@pytest.mark.parametrize("keyword_only", (False, True))
def test_first_header_operand_does_not_require_a_later_default(keyword_only):
    parameters = "first=object, second=absent"
    if keyword_only:
        parameters = "*, " + parameters
    text = f"def unused({parameters}): pass\n"
    with pytest.raises(NameError, match="absent"):
        runtime(text)
    environment = execution(text)
    function = environment.module.module.body[0]
    defaults = function.args.kw_defaults if keyword_only else function.args.defaults
    environment.capture(defaults[0]).require_native_identity(NativeDeclaration(object))
    with pytest.raises(ValueError, match="unproved"):
        environment.capture(defaults[1]).require_closed()
    assert not environment._pending


@pytest.mark.parametrize(
    "definition",
    ("def unused(value=object): pass", "class Unused(object): pass"),
)
def test_missing_decorator_prevents_later_header_operands(definition):
    text = "@absent\n" + definition + "\n"
    with pytest.raises(NameError, match="absent"):
        runtime(text)
    environment = execution(text)
    node = environment.module.module.body[0]
    operand = (
        node.args.defaults[0] if isinstance(node, ast.FunctionDef) else node.bases[0]
    )
    with pytest.raises(ValueError, match="unproved"):
        environment.capture(operand).require_native_identity(NativeDeclaration(object))
    assert not environment._pending


def test_missing_first_class_base_prevents_later_base_read():
    text = "class Unused(absent, object): pass\n"
    with pytest.raises(NameError, match="absent"):
        runtime(text)
    environment = execution(text)
    node = environment.module.module.body[0]
    with pytest.raises(ValueError, match="unproved"):
        environment.capture(node.bases[-1]).require_native_identity(
            NativeDeclaration(object)
        )
    assert not environment._pending


@pytest.mark.parametrize(
    "definition",
    (
        "def unused(): return absent",
        "async def unused(): return absent",
        "class Unused: pass",
    ),
)
def test_empty_header_has_no_unresolved_evaluation_obligation(definition):
    text = definition + "\nlater = object\n"
    assert runtime(text)["later"] is object
    environment = execution(text)
    require_endpoint(environment, "prefix")
    require_endpoint(environment, "later_read")
    assert not environment._pending


@pytest.mark.parametrize(
    "deferred_definition",
    ("def inner(value=absent): pass", "class Inner(absent): pass"),
)
def test_header_inside_uninvoked_function_remains_in_its_own_activation(
    deferred_definition,
):
    text = f"def outer():\n    {deferred_definition}\nlater = object\n"
    namespace = runtime(text)
    environment = execution(text)
    require_endpoint(environment, "prefix")
    require_endpoint(environment, "later_read")
    with pytest.raises(NameError, match="absent"):
        namespace["outer"]()
    assert not environment._pending


def header_effect(environment, node):
    return next(
        (site, effect)
        for site in environment.effects.sites
        if site.trigger is node
        for effect in site.effects
        if isinstance(effect, DefinitionHeaderEffect)
    )


def test_header_effect_retains_actual_ordered_receipts_and_no_whole_effect_fallback():
    environment = execution("def unused(first=object, *, second=type): pass\n")
    node = environment.module.module.body[0]
    site, effect = header_effect(environment, node)
    target = environment.source.definition_operation(node).event.target
    operations = effect.application_operations(site, environment.source)
    assert tuple(operation.event for operation in operations) == target.input_uses
    assert all(
        environment.source.event_operation(operation.event) is operation
        for operation in operations
    )
    assert tuple(operation.node for operation in operations) == (
        node.args.defaults[0],
        node.args.kw_defaults[0],
    )
    for operation in operations:
        effect.require_operation(environment, operation)
    with pytest.raises(ValueError, match="original input operations"):
        effect.require_closed(environment)


@pytest.mark.parametrize(
    "foreign", ("copied_operation", "other_definition", "other_source")
)
def test_header_effect_refuses_receipts_without_its_original_definition_provenance(
    foreign,
):
    text = "def first(value=object): pass\ndef second(value=type): pass\n"
    environment = execution(text)
    node = environment.module.module.body[0]
    site, effect = header_effect(environment, node)
    (original,) = effect.application_operations(site, environment.source)
    effect.require_operation(environment, original)
    if foreign == "copied_operation":
        selected = replace(original)
        assert selected.node is original.node
        assert selected.owner is original.owner
        assert selected.event is original.event
    elif foreign == "other_definition":
        second = environment.module.module.body[1]
        second_site, second_effect = header_effect(environment, second)
        (selected,) = second_effect.application_operations(
            second_site, environment.source
        )
    else:
        other = execution(text)
        other_site, other_effect = header_effect(other, other.module.module.body[0])
        (selected,) = other_effect.application_operations(other_site, other.source)
    assert selected is not original
    with pytest.raises(ValueError):
        effect.require_operation(environment, selected)


def test_empty_header_projection_is_exhaustive_without_a_synthetic_occurrence():
    environment = execution("def unused(): pass\n")
    node = environment.module.module.body[0]
    site, effect = header_effect(environment, node)
    assert effect.application_operations(site, environment.source) == ()
    prefix = require_endpoint(environment, "prefix")
    assert not any(
        isinstance(occurrence.effect, DefinitionHeaderEffect)
        for interval in prefix.intervals
        for occurrence in environment.effects.occurrences(environment.source, interval)
    )
    with pytest.raises(ValueError, match="original input operations"):
        effect.require_closed(environment)
