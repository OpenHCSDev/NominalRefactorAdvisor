"""Function binding determination belongs to the shared source/scope owner."""

import ast
from collections import Counter
from contextlib import contextmanager

import pytest

from nominal_refactor_advisor.ast_tools import (
    ParsedModuleClassFunctionStackNodeVisitor,
    SourceModule,
)
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.record_checks import DeclaredTypeCheckModule
from nominal_refactor_advisor.semantic_descent import (
    PresentationProjectionKind,
    _CompactSemanticProjectionVisitor,
    _ProjectionVisitor,
)


SOURCE = """\
def outer(value: Value, payload):
    local = payload['left']
    def nested(value: Value):
        other = 1
        return isinstance(value.count, int)
    async def nested_async(value: Value):
        return isinstance(value.count, int)
    return payload.get('right')

def shadowed(value: Value, isinstance):
    return isinstance(value.count, int)
"""


class BindingCensus(ParsedModuleClassFunctionStackNodeVisitor):
    def __init__(self, parsed_module, **kwargs):
        super().__init__(parsed_module, **kwargs)
        self.binding_events = []

    @contextmanager
    def function_scope(self, node, assigned_names, argument_names):
        self.binding_events.append(
            ("enter", node.name, self.qualname, assigned_names, argument_names)
        )
        try:
            with super().function_scope(node, assigned_names, argument_names):
                yield
        finally:
            self.binding_events.append(("exit", node.name, self.qualname))


class CensusBeforeProjection(BindingCensus, _CompactSemanticProjectionVisitor):
    pass


class CensusAfterProjection(_CompactSemanticProjectionVisitor, BindingCensus):
    pass


@pytest.mark.parametrize("visitor_type", (CensusBeforeProjection, CensusAfterProjection))
def test_shared_function_bindings_need_one_determination_per_event(
    tmp_path, monkeypatch, visitor_type
):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    functions = tuple(
        node
        for node in ast.walk(module.module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    body_ids = {id(node.body) for node in functions}
    body_calls = Counter()
    argument_calls = Counter()
    original_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names
    original_arguments = LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names

    def observed_bound(nodes):
        if id(nodes) in body_ids:
            body_calls[id(nodes)] += 1
        return original_bound(nodes)

    def observed_arguments(node):
        argument_calls[id(node)] += 1
        return original_arguments(node)

    # Parse/source identity and existing lexical algorithm are unchanged. Only
    # observe calls during this one original generic visitor invocation.
    monkeypatch.setattr(LEXICAL_SCOPE_BINDING_AUTHORITY, "bound_names", observed_bound)
    monkeypatch.setattr(
        LEXICAL_SCOPE_BINDING_AUTHORITY, "argument_names", observed_arguments
    )
    visitor = visitor_type(module)
    visitor.visit(module.module)
    assert body_calls == Counter({id(node.body): 1 for node in functions})
    assert argument_calls == Counter({id(node): 1 for node in functions})
    assert visitor.binding_events == [
        (
            "enter", "outer", "<module>",
            frozenset(("local", "nested", "nested_async")),
            frozenset(("value", "payload")),
        ),
        ("enter", "nested", "outer", frozenset(("other",)), frozenset(("value",))),
        ("exit", "nested", "outer"),
        ("enter", "nested_async", "outer", frozenset(), frozenset(("value",))),
        ("exit", "nested_async", "outer"),
        ("exit", "outer", "<module>"),
        (
            "enter", "shadowed", "<module>", frozenset(),
            frozenset(("value", "isinstance")),
        ),
        ("exit", "shadowed", "<module>"),
    ]
    independent = _ProjectionVisitor(module)
    independent.visit(module.module)
    assert visitor.projections == independent.projections
    assert visitor.class_supplements == independent.class_supplements
    assert DeclaredTypeCheckModule.from_collector(
        visitor
    ) == DeclaredTypeCheckModule.collect(module)
    assert len(visitor.checks) == 2
    assert visitor_type.__mro__.count(ParsedModuleClassFunctionStackNodeVisitor) == 1
    assert visitor.class_stack == visitor.function_stack == []
    assert visitor.subjects == visitor.locals == []
    assert visitor.type_scopes == visitor.owner_construction_stack == []


@pytest.mark.parametrize("visitor_type", (CensusBeforeProjection, CensusAfterProjection))
def test_function_scope_recomputes_each_invocation_without_a_parallel_cache(
    tmp_path, visitor_type
):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    visitor = visitor_type(module)
    visitor.visit(module.module)
    # The determining AST is intentionally changed to distinguish scoped reuse
    # from a stale node-id cache. Nothing is persisted or relabeled as new source.
    outer = module.module.body[0]
    outer.body.insert(0, ast.parse("fresh = 1").body[0])
    before = len(visitor.binding_events)
    visitor.visit(module.module)
    event = visitor.binding_events[before]
    assert event[:3] == ("enter", "outer", "<module>")
    assert event[3] == frozenset(("fresh", "local", "nested", "nested_async"))


@pytest.mark.parametrize("visitor_type", (CensusBeforeProjection, CensusAfterProjection))
def test_function_scope_body_failure_unwinds_cooperative_contexts(tmp_path, visitor_type):
    class FailOnCall(ParsedModuleClassFunctionStackNodeVisitor):
        def visit_Call(self, node):
            raise RuntimeError("controlled binding-scope body failure")

    class Failing(visitor_type, FailOnCall):
        pass

    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    visitor = Failing(module)
    with pytest.raises(RuntimeError, match="controlled binding-scope body failure"):
        visitor.visit(module.module)
    assert visitor.binding_events[-2:] == [
        ("exit", "nested", "outer"), ("exit", "outer", "<module>")
    ]
    assert visitor.class_stack == visitor.function_stack == []
    assert visitor.subjects == visitor.locals == []
    assert visitor.type_scopes == visitor.owner_construction_stack == []
    assert visitor.class_supplement_stack == visitor.active_class_method_frames == []
    assert visitor._projection_suppression_depth == 0


class FailScopeEntry(ParsedModuleClassFunctionStackNodeVisitor):
    @contextmanager
    def function_scope(self, node, assigned_names, argument_names):
        raise RuntimeError("controlled binding-scope entry failure")
        yield


class FailScopeExit(ParsedModuleClassFunctionStackNodeVisitor):
    @contextmanager
    def function_scope(self, node, assigned_names, argument_names):
        with super().function_scope(node, assigned_names, argument_names):
            yield
        raise RuntimeError("controlled binding-scope exit failure")


class EntryFailureBeforeProjection(FailScopeEntry, _CompactSemanticProjectionVisitor):
    pass


class EntryFailureAfterProjection(_CompactSemanticProjectionVisitor, FailScopeEntry):
    pass


class ExitFailureBeforeProjection(FailScopeExit, _CompactSemanticProjectionVisitor):
    pass


class ExitFailureAfterProjection(_CompactSemanticProjectionVisitor, FailScopeExit):
    pass


@pytest.mark.parametrize(
    "visitor_type",
    (
        EntryFailureBeforeProjection,
        EntryFailureAfterProjection,
        ExitFailureBeforeProjection,
        ExitFailureAfterProjection,
    ),
)
def test_function_scope_entry_and_exit_failures_unwind_in_both_mro_orders(
    tmp_path, visitor_type
):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    visitor = visitor_type(module)
    with pytest.raises(RuntimeError, match="controlled binding-scope .* failure"):
        visitor.visit(module.module)
    assert visitor.class_stack == visitor.function_stack == []
    assert visitor.subjects == visitor.locals == []
    assert visitor.type_scopes == visitor.owner_construction_stack == []
    assert visitor.class_supplement_stack == visitor.active_class_method_frames == []
    assert visitor._projection_suppression_depth == 0


@pytest.mark.parametrize("visitor_type", (CensusBeforeProjection, CensusAfterProjection))
def test_mapping_postlude_failure_uses_the_shared_name_scope_owner(
    tmp_path, monkeypatch, visitor_type
):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    original = _ProjectionVisitor._append_projection

    def controlled_failure(self, node, kind, *args, **kwargs):
        if kind is PresentationProjectionKind.MAPPING_READ:
            assert self.qualname == "outer"
            raise RuntimeError("controlled mapping postlude failure")
        return original(self, node, kind, *args, **kwargs)

    monkeypatch.setattr(_ProjectionVisitor, "_append_projection", controlled_failure)
    visitor = visitor_type(module)
    with pytest.raises(RuntimeError, match="controlled mapping postlude failure"):
        visitor.visit(module.module)
    assert visitor.class_stack == visitor.function_stack == []
    assert visitor.subjects == visitor.locals == []
    assert visitor.type_scopes == visitor.owner_construction_stack == []
    assert visitor.binding_events[-1] == ("exit", "outer", "<module>")
