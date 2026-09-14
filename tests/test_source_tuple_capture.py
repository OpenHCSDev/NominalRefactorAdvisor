"""Original tuple construction retains ordered inputs, not general equivalence."""

import ast
from copy import deepcopy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.manual_registry import DirectManualRegistryComponent
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.product_flow import (
    CompactBindingVisit,
    CompactTupleValue,
    CompactValueResolverABC,
)
from nominal_refactor_advisor.source_execution import SourceTupleCapture
from test_source_distinct_item_stores import BASE, execution


def tuple_node(environment):
    return next(
        node.value
        for node in environment.module.module.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
    )


def tuple_capture(environment):
    result = environment.capture_value(tuple_node(environment))
    assert isinstance(result, SourceTupleCapture)
    result.require_closed()
    return result


def producer_node(environment, value):
    context, binding = value.source_definition()
    return environment.source_operation(context, binding).node


def authored_runtime(source):
    namespace = {}
    exec(compile(source, "<authored-source-tuple>", "exec"), namespace)
    return namespace


class ConservativeResolver(CompactValueResolverABC):
    def _unproved_value_resolution(self, context):
        return context

    def _lexical_value_resolution(self, reference, context):
        return context

    _compiler_stored_value_resolution = _lexical_value_resolution
    _forwarded_result_value_resolution = _lexical_value_resolution
    _call_result_value_resolution = _lexical_value_resolution


def test_compact_tuple_retains_actual_ordered_element_productions():
    environment = execution(BASE + "held = (Alpha, Beta)\n")
    node = tuple_node(environment)
    read = environment.source.value_reads_by_node[node]
    production = read.use.value
    assert isinstance(production, CompactTupleValue)
    assert len(production.inputs) == 2
    for element, use in zip(node.elts, production.inputs, strict=True):
        assert environment.source.value_reads_by_node[element].use is use
        assert environment.source_operation(read.context, use).node is element
        assert use.position.dominates(read.use.position)
    assert production.inputs[0].position.dominates(production.inputs[1].position)
    unknown = object()
    assert production.resolve_value(ConservativeResolver(), unknown) is unknown


def test_source_tuple_elements_keep_actual_original_class_creation_owners():
    source = BASE + "held = (Alpha, Beta)\nalias = held\n"
    environment = execution(source)
    captured = tuple_capture(environment)
    assert captured.native_type is tuple
    alpha, beta = environment.module.module.body[1:3]
    assert producer_node(environment, captured.elements[0]) is alpha
    assert producer_node(environment, captured.elements[1]) is beta
    alias = environment.capture_value(environment.module.module.body[-1].value)
    alias.require_closed()
    assert isinstance(alias, SourceTupleCapture)
    assert alias.read.use is captured.read.use
    runtime = authored_runtime(source)
    assert runtime["alias"] is runtime["held"]
    assert runtime["held"] == (runtime["Alpha"], runtime["Beta"])


def test_nested_tuple_members_derive_from_their_own_original_productions():
    source = BASE + "held = (Alpha, (Beta, Alpha))\n"
    environment = execution(source)
    captured = tuple_capture(environment)
    inner = captured.elements[1]
    assert isinstance(inner, SourceTupleCapture)
    assert (
        inner.read.use
        is environment.source.value_reads_by_node[tuple_node(environment).elts[1]].use
    )
    alpha, beta = environment.module.module.body[1:3]
    assert producer_node(environment, captured.elements[0]) is alpha
    assert producer_node(environment, inner.elements[0]) is beta
    assert producer_node(environment, inner.elements[1]) is alpha
    runtime = authored_runtime(source)
    assert runtime["held"][1] == (runtime["Beta"], runtime["Alpha"])


def test_duplicate_values_are_distinct_input_events_not_duplicate_metadata():
    environment = execution(BASE + "held = (Alpha, Alpha)\n")
    captured = tuple_capture(environment)
    inputs = captured.read.use.value.inputs
    assert inputs[0] is not inputs[1]
    assert producer_node(environment, captured.elements[0]) is producer_node(
        environment, captured.elements[1]
    )


def test_alias_after_rebind_retains_historical_tuple_members():
    source = BASE + "saved = Alpha\nheld = (Alpha,)\nAlpha = Beta\ntail = held\n"
    environment = execution(source)
    captured = tuple_capture(environment)
    tail = environment.capture_value(environment.module.module.body[-1].value)
    tail.require_closed()
    assert tail.read.use is captured.read.use
    assert (
        producer_node(environment, tail.elements[0])
        is environment.module.module.body[1]
    )
    runtime = authored_runtime(source)
    assert runtime["tail"][0] is runtime["saved"]
    assert runtime["tail"][0] is not runtime["Alpha"]


def test_ordered_walrus_uses_each_original_capture_cut():
    source = BASE + "saved = Alpha\nheld = (Alpha, (Alpha := Beta), Alpha)\n"
    environment = execution(source)
    captured = tuple_capture(environment)
    alpha, beta = environment.module.module.body[1:3]
    assert tuple(producer_node(environment, value) for value in captured.elements) == (
        alpha,
        beta,
        beta,
    )
    runtime = authored_runtime(source)
    assert runtime["held"] == (runtime["saved"], runtime["Beta"], runtime["Beta"])


def test_retained_class_tuple_does_not_block_original_registry_entry_proof():
    source = (
        BASE
        + "ALL_HANDLERS = (Alpha, Beta)\nREGISTRY['alpha'] = Alpha\nREGISTRY['beta'] = Beta\n"
    )
    environment = execution(source)
    tuple_capture(environment)
    component = DirectManualRegistryComponent.from_module_anchor(
        environment.module.module, "Alpha"
    )
    component.require_original_entry_values(environment)
    runtime = authored_runtime(source)
    assert runtime["ALL_HANDLERS"] == tuple(runtime["REGISTRY"].values())


@pytest.mark.parametrize(
    "defect", ("copied_use", "foreign_context", "foreign_execution")
)
def test_tuple_capture_rejects_noncanonical_source_admission(defect):
    environment = execution(BASE + "held = (Alpha, Beta)\n")
    read = environment.source.value_reads_by_node[tuple_node(environment)]
    other = execution(environment.module.source)
    if defect == "copied_use":
        read = replace(read, use=replace(read.use))
    elif defect == "foreign_context":
        read = replace(read, context=other.entry.context)
    else:
        environment = other
    with pytest.raises(ValueError):
        SourceTupleCapture(environment, read, frozenset()).require_closed()


@pytest.mark.parametrize("defect", ("reordered", "duplicated", "omitted"))
def test_tuple_inputs_must_match_actual_source_elements_and_order(defect):
    environment = execution(BASE + "held = (Alpha, Beta)\n")
    read = environment.source.value_reads_by_node[tuple_node(environment)]
    production = read.use.value
    inputs = production.inputs
    if defect == "reordered":
        invalid = tuple(reversed(inputs))
    elif defect == "duplicated":
        invalid = (inputs[0], inputs[0])
    else:
        invalid = inputs[:1]
    # Corrupt the actual nominal input association, without replacing the
    # canonical outer event: geometry/outer identity alone cannot authorize it.
    object.__setattr__(production, "inputs", invalid)
    with pytest.raises(ValueError):
        SourceTupleCapture(environment, read, frozenset()).require_closed()


def test_copied_tuple_ast_is_not_an_original_capture():
    environment = execution(BASE + "held = (Alpha, Beta)\n")
    with pytest.raises(ValueError):
        environment.capture_value(deepcopy(tuple_node(environment))).require_closed()


@pytest.mark.parametrize(
    "expression", ("(*unknown,)", "(Alpha, missing)", "(unknown(), Alpha)")
)
def test_unproved_element_or_unpacking_cannot_gain_tuple_admission(expression):
    environment = execution(BASE + f"held = {expression}\n")
    captured = environment.capture_value(tuple_node(environment))
    assert isinstance(captured, OpenCapturedReference)
    with pytest.raises(ValueError):
        captured.require_closed()


@pytest.mark.parametrize(
    "source",
    (
        "if flag:\n    held=(object,)\n",
        "for item in items:\n    held=(object,)\n",
        "def deferred():\n    held=(object,)\n",
    ),
)
def test_conditional_repeated_or_unproved_activation_remains_open(source):
    environment = execution(source)
    node = next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Tuple)
    )
    captured = environment.capture_value(node)
    assert isinstance(captured, OpenCapturedReference)
    with pytest.raises(ValueError):
        captured.require_closed()


def test_unsupported_native_backend_does_not_gain_tuple_construction(monkeypatch):
    environment = execution("held=(object,)\n")
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    captured = environment.capture_value(tuple_node(environment))
    assert isinstance(captured, OpenCapturedReference)
    with pytest.raises(ValueError):
        captured.require_closed()


def test_tuple_construction_does_not_grant_inert_release():
    environment = execution(BASE + "held=(Alpha,)\nheld=None\ntail=Alpha\n")
    captured = tuple_capture(environment)
    with pytest.raises(ValueError):
        captured.require_release()
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[-1].value
        ).require_closed()


def test_tuple_construction_does_not_require_or_invoke_element_hashing():
    source = (
        "class Item:\n"
        "    def __hash__(self):\n"
        "        raise RuntimeError('hashing is not tuple construction')\n"
        "item=Item()\nheld=(item,)\n"
    )
    environment = execution(source)
    original_call = environment.module.module.body[1].value
    original = environment.capture_value(original_call)
    original.require_closed()
    captured = tuple_capture(environment)
    assert captured.elements[0] is original
    runtime = authored_runtime(source)
    assert runtime["held"][0] is runtime["item"]
    with pytest.raises(RuntimeError, match="not tuple construction"):
        hash(runtime["held"])


def test_starred_source_iterator_is_not_ordinary_element_retention():
    source = (
        "class Source:\n"
        "    def __iter__(self):\n"
        "        raise RuntimeError('iteration was invoked')\n"
        "held=(*Source(),)\n"
    )
    environment = execution(source)
    with pytest.raises(ValueError):
        environment.capture_value(tuple_node(environment)).require_closed()
    with pytest.raises(RuntimeError, match="iteration was invoked"):
        authored_runtime(source)


def test_tuple_element_capture_preserves_the_pending_alias_cycle_guard():
    environment = execution("class Alpha:pass\nalias=Alpha\nheld=(alias,)\n")
    tuple_capture(environment)
    read = environment.source.value_reads_by_node[tuple_node(environment)]
    alias = next(
        mutation
        for mutation in read.context.flow.mutations
        if mutation.target.bound_name == "alias"
    )
    pending = frozenset((CompactBindingVisit(read.context, alias),))
    with pytest.raises(CapturedReferenceRejection) as caught:
        SourceTupleCapture(environment, read, pending).require_closed()
    assert caught.value.violation is CapturedReferenceViolation.CYCLIC_BINDING
