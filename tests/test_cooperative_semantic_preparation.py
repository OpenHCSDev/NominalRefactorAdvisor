"""Independent semantic capabilities compose without a second AST traversal."""

import ast
from collections import Counter
from dataclasses import replace

import pytest

from nominal_refactor_advisor.ast_tools import (
    ParsedModuleClassFunctionStackNodeVisitor,
    SourceModule,
    retains_python_ast,
)
from nominal_refactor_advisor.record_checks import (
    DeclaredAttributeCheckCollector,
    DeclaredTypeCheckModule,
)
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticModuleProjectionFamily,
    _CompactSemanticProjectionVisitor,
    _ProjectionVisitor,
)

SOURCE = """\
class Value:
    count: int

class Example:
    def method(self, value: Value, payload):
        if isinstance(value.count, int):
            pass
        type(value.count) is int
        ignored = lambda: isinstance(value.count, str)
        data = {'value': payload['value'], 'count': payload.get('count')}
        def nested(other: Value):
            return isinstance(other.count, int)
        return data

async def async_function(value: Value):
    return isinstance(value.count, int)

def shadowed(value: Value, isinstance):
    return isinstance(value.count, int)

def reassigned(value: Value):
    value = None
    return isinstance(value.count, int)
"""


@pytest.mark.parametrize(
    "suffix",
    (
        "",
        "\nisinstance = lambda *args: True\n",
        "\ntype = lambda value: int\n",
    ),
)
def test_composed_facts_equal_independent_capabilities(tmp_path, suffix):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE + suffix).parse()
    independent = _ProjectionVisitor(module, None)
    independent.visit(module.module)
    checks = DeclaredTypeCheckModule.collect(module)
    composed = _CompactSemanticProjectionVisitor(module, None)
    composed.visit(module.module)
    assert composed.projections == independent.projections
    assert composed.class_supplements == independent.class_supplements
    assert DeclaredTypeCheckModule.from_collector(composed) == checks
    assert not retains_python_ast(tuple(composed.projections))
    assert not retains_python_ast(checks)
    assert composed.class_stack == composed.function_stack == []
    assert composed.subjects == composed.locals == []
    assert composed.parsed_module is module
    if not suffix:
        assert len(checks.checks) == 4
        assert tuple(check.location.symbol for check in checks.checks) == (
            "Example.method",
            "Example.method",
            "Example.method.nested",
            "async_function",
        )
        assert all("str" not in check.expression for check in checks.checks)


def test_compact_family_does_not_reenter_independent_check_collection(
    tmp_path, monkeypatch
):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()

    def forbidden(*args, **kwargs):
        raise AssertionError("The complete family must not traverse checks again")

    monkeypatch.setattr(DeclaredTypeCheckModule, "collect", forbidden)
    (projection,) = CompactSemanticModuleProjectionFamily.collect(module)
    assert len(projection.type_checks.checks) == 4
    assert projection.projections
    assert not retains_python_ast(projection)


def test_new_capability_is_only_a_declaration_and_cooperative_hook(tmp_path):
    class CallCensus(ParsedModuleClassFunctionStackNodeVisitor):
        def __init__(self, parsed_module):
            super().__init__(parsed_module)
            self.calls = Counter()

        def visit_Call(self, node):
            self.calls[id(node)] += 1
            super().visit_Call(node)

    class Extended(CallCensus, _CompactSemanticProjectionVisitor):
        pass

    # All constructors and event handlers reach the original shared owner once.
    # The new capability has no taxonomy registration or generic consumer edit.
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    visitor = Extended(module)
    visitor.visit(module.module)
    assert visitor.calls == Counter(
        {id(node): 1 for node in ast.walk(module.module) if isinstance(node, ast.Call)}
    )
    assert DeclaredTypeCheckModule.from_collector(
        visitor
    ) == DeclaredTypeCheckModule.collect(module)
    mro = Extended.__mro__
    assert mro.index(_ProjectionVisitor) < mro.index(DeclaredAttributeCheckCollector)
    assert mro.count(ParsedModuleClassFunctionStackNodeVisitor) == 1


def test_exception_unwinds_both_capabilities_and_shared_owner(tmp_path):
    class FailOnCall(ParsedModuleClassFunctionStackNodeVisitor):
        def visit_Call(self, node):
            raise RuntimeError("controlled syntax failure")

    class Failing(_CompactSemanticProjectionVisitor, FailOnCall):
        pass

    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    visitor = Failing(module, None)
    with pytest.raises(RuntimeError, match="controlled syntax failure"):
        visitor.visit(module.module)
    assert visitor.class_stack == visitor.function_stack == []
    assert visitor.subjects == visitor.locals == []
    assert visitor.owner_construction_stack == visitor.type_scopes == []
    assert visitor.class_supplement_stack == visitor.active_class_method_frames == []
    assert visitor._projection_suppression_depth == 0


@pytest.mark.parametrize(
    "statement",
    (
        "result = {'left': isinstance(value.count, int), 'right': 'name'}",
        "result: dict = {'left': isinstance(value.count, int), 'right': 'name'}",
        "return {'left': isinstance(value.count, int), 'right': 'name'}",
    ),
)
def test_presentation_short_circuit_does_not_hide_another_capability(
    tmp_path, statement
):
    source = "def target(value: Value):\n    " + statement + "\n"
    module = SourceModule(tmp_path / "case.py", "case", source).parse()
    independent = _ProjectionVisitor(module, None)
    independent.visit(module.module)
    composed = _CompactSemanticProjectionVisitor(module, None)
    composed.visit(module.module)
    assert composed.projections == independent.projections
    checks = DeclaredTypeCheckModule.from_collector(composed)
    assert checks == DeclaredTypeCheckModule.collect(module)
    assert len(checks.checks) == 1
    assert composed._projection_suppression_depth == 0


def test_context_only_family_keeps_its_existing_no_checks_contract(tmp_path):
    module = SourceModule(tmp_path / "case.py", "case", SOURCE).parse()
    demand = CompactSemanticModuleProjectionFamily.report_demand((), None)
    (focused,) = CompactSemanticModuleProjectionFamily.collect_demanded(module, demand)
    (complete,) = CompactSemanticModuleProjectionFamily.collect(module)
    assert focused.type_checks == replace(complete.type_checks, checks=())
    assert focused.projections == ()
    assert focused.class_supplements == complete.class_supplements
