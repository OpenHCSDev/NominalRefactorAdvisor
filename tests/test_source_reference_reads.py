"""Native gate inputs retain an actual module-qualified source read."""

import ast
from dataclasses import replace
from pathlib import Path
import pickle
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def _projection(source, name="reads"):
    module = ParsedModule(Path(name + ".py"), name, False, ast.parse(source), source)
    return module, compact_product_flow_projection(module)


def test_exact_read_join_retains_callee_capture_before_argument_effects():
    module, projection = _projection("callback(argument())\n")
    outer = module.module.body[0].value
    read = projection.reference_reads_by_span[SourceByteSpan.require_node(outer.func)]
    outer_call = next(
        call
        for call in read.context.flow.calls
        if call.target.terminal_name == "callback"
    )
    argument_call = next(
        call
        for call in read.context.flow.calls
        if call.target.terminal_name == "argument"
    )
    assert read.context is projection.flow_contexts[0]
    assert read.use is outer_call.target_use
    assert read.use.position.dominates(argument_call.position)
    assert argument_call.position.dominates(outer_call.position)


def test_same_source_span_in_two_modules_keeps_separate_contexts():
    module, first = _projection("read\n", "first")
    _, second = _projection("read\n", "second")
    span = SourceByteSpan.require_node(module.module.body[0].value)
    assert first.reference_reads_by_span[span].context.module_name == "first"
    assert second.reference_reads_by_span[span].context.module_name == "second"
    assert (
        first.reference_reads_by_span[span].use
        is not second.reference_reads_by_span[span].use
    )


def test_repeated_declaration_names_keep_the_read_in_its_actual_body():
    module, projection = _projection("class Owner: first\nclass Owner: second\n")
    first, second = module.module.body
    reads = [
        projection.reference_reads_by_span[
            SourceByteSpan.require_node(owner.body[0].value)
        ]
        for owner in (first, second)
    ]
    assert reads[0].context.flow.owner != reads[1].context.flow.owner
    assert reads[0].context.flow.owner.qualname == reads[1].context.flow.owner.qualname
    assert reads[0].context.flow.owner.source_span == SourceByteSpan.require_node(first)
    assert reads[1].context.flow.owner.source_span == SourceByteSpan.require_node(
        second
    )


def test_duplicate_read_handle_is_unproved_without_selecting_first():
    module, projection = _projection("read\n")
    flow = projection.flows[0]
    duplicated = replace(flow, callable_reference_uses=flow.callable_reference_uses * 2)
    changed = replace(projection, flows=(duplicated,))
    span = SourceByteSpan.require_node(module.module.body[0].value)
    assert span not in changed.reference_reads_by_span


def test_computed_subscription_origin_cannot_borrow_a_child_read():
    module, projection = _projection("(list if condition else dict)[str]\n")
    origin = module.module.body[0].value.value
    assert SourceByteSpan.require_node(origin) not in projection.reference_reads_by_span
    for child in (origin.test, origin.body, origin.orelse):
        assert SourceByteSpan.require_node(child) in projection.reference_reads_by_span


def test_class_base_reads_belong_to_enclosing_flow_not_body():
    module, projection = _projection("class Owner(Generic[T]): body\n")
    owner = module.module.body[0]
    base_read = projection.reference_reads_by_span[
        SourceByteSpan.require_node(owner.bases[0].value)
    ]
    body_read = projection.reference_reads_by_span[
        SourceByteSpan.require_node(owner.body[0].value)
    ]
    assert base_read.context.flow.owner.kind.is_module_scope
    assert body_read.context.flow.owner.kind.is_class_body_scope


def test_read_context_and_receipt_sharing_survive_populated_pickle():
    _, projection = _projection("class Owner: callback(value)\n")
    before = projection.reference_reads_by_span
    restored = pickle.loads(pickle.dumps(projection))
    assert before.keys() == restored.reference_reads_by_span.keys()
    for read in restored.reference_reads_by_span.values():
        assert any(read.context is context for context in restored.flow_contexts)
        assert any(read.use is use for use in read.context.flow.reference_uses)


@pytest.mark.parametrize("scope", ("module", "class", "function"))
@pytest.mark.parametrize("future", (False, True))
@pytest.mark.parametrize("assignment", (False, True))
@pytest.mark.parametrize(
    "target", ("field", "(field)", "receiver().field", "receiver()[index()]")
)
def test_variable_annotation_reads_match_native_scope_phase_and_order(
    scope, future, assignment, target
):
    statement = target + ": annotation()" + (" = value()" if assignment else "") + "\n"
    prefixes = {"module": "", "class": "class Owner:\n", "function": "def body():\n"}
    source = ("from __future__ import annotations\n" if future else "") + prefixes[
        scope
    ]
    source += ("    " if scope != "module" else "") + statement
    if scope == "function":
        source += "body()\n"
    events = []

    class Receiver:
        def __setattr__(self, name, value):
            events.append("write")

        def __setitem__(self, key, value):
            events.append("write")

    def receiver():
        events.append("receiver")
        return Receiver()

    def index():
        events.append("index")
        return 0

    def annotation():
        events.append("annotation")
        return int

    def value():
        events.append("value")
        return 1

    exec(
        compile(source, "<trusted-annotation-order>", "exec", dont_inherit=True),
        {
            "receiver": receiver,
            "index": index,
            "annotation": annotation,
            "value": value,
        },
    )
    eager = not future and sys.version_info < (3, 14) and scope != "function"
    assert ("annotation" in events) is eager
    _, projection = _projection(source)
    owner_name = {"module": "", "class": "Owner", "function": "body"}[scope]
    flow = next(flow for flow in projection.flows if flow.owner.qualname == owner_name)
    assert [call.target.terminal_name for call in flow.calls] == [
        event for event in events if event != "write"
    ]
    annotation_calls = [
        call for call in flow.calls if call.target.terminal_name == "annotation"
    ]
    if eager and assignment:
        (annotation_call,) = annotation_calls
        (assignment_write,) = flow.mutations
        assert assignment_write.position.dominates(annotation_call.target_use.position)


def test_eager_nested_annotation_names_have_actual_class_read_receipts():
    module, projection = _projection(
        "class Owner:\n    field: CV[dict[str, tuple[int, ...]]] = 1\n"
    )
    annotation = module.module.body[0].body[0].annotation
    names = [node for node in ast.walk(annotation) if isinstance(node, ast.Name)]
    assert len(names) == 5
    for name in names:
        span = SourceByteSpan.require_node(name)
        if sys.version_info < (3, 14):
            assert (
                projection.reference_reads_by_span[span].context.flow.owner.qualname
                == "Owner"
            )
        else:
            assert span not in projection.reference_reads_by_span


@pytest.mark.parametrize("prefix", ("def", "async def"))
@pytest.mark.parametrize("future", (False, True))
def test_function_annotations_do_not_manufacture_enclosing_lazy_reads(prefix, future):
    source = ("from __future__ import annotations\n" if future else "") + (
        f"{prefix} function(arg: annotation() = default()) -> result(): pass\n"
    )
    events = []

    def annotation():
        events.append("annotation")
        return int

    def default():
        events.append("default")
        return None

    def result():
        events.append("result")
        return str

    namespace = {"annotation": annotation, "default": default, "result": result}
    exec(
        compile(source, "<trusted-header-phase>", "exec", dont_inherit=True), namespace
    )
    _, projection = _projection(source)
    assert [call.target.terminal_name for call in projection.flows[0].calls] == events
    # Source metadata remains available even where reading it would evaluate later.
    assert (
        projection.function_declarations[0].return_annotation_expression == "result()"
    )
