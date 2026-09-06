"""Shared default traversal separates declaration effects from deferred bodies."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import EagerNameLoadCollector, ParsedModule
from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyUse,
    FunctionParameterBinding,
    ModuleBindingResolutionPhase,
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.lexical_bindings import FunctionDefaultVisitor
from nominal_refactor_advisor.product_flow import compact_product_flow_projection


def _flow(source: str, owner: str = ""):
    module = ParsedModule(
        path=Path("declaration_defaults.py"),
        module_name="declaration_defaults",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return next(
        flow
        for flow in compact_product_flow_projection(module).flows
        if flow.owner.qualname == owner
    )


def _call_names(flow) -> tuple[str, ...]:
    references = tuple(call.target_use.target.lexical_reference for call in flow.calls)
    assert all(reference is not None for reference in references)
    return tuple(reference.root_name for reference in references)


@pytest.mark.parametrize(
    "source, owner",
    [
        (
            "def sample(a=first(), /, b=second(), *items, required, "
            "c=third(), d=fourth(), **options): deferred()",
            "",
        ),
        (
            "async def sample(a=first(), /, b=second(), *items, required, "
            "c=third(), d=fourth(), **options): deferred()",
            "",
        ),
        (
            "stored = lambda a=first(), /, b=second(), *items, required, "
            "c=third(), d=fourth(), **options: deferred()",
            "",
        ),
        (
            "class Owner:\n"
            "    stored = lambda a=first(), /, b=second(), *items, required, "
            "c=third(), d=fourth(), **options: deferred()",
            "Owner",
        ),
        (
            "def container():\n"
            "    return lambda a=first(), /, b=second(), *items, required, "
            "c=third(), d=fourth(), **options: deferred()\n"
            "container()",
            "container",
        ),
    ],
)
def test_compact_default_order_matches_trusted_native_execution(
    source: str, owner: str
) -> None:
    events = []

    def record(name):
        def called():
            events.append(name)
            return None

        return called

    namespace = {
        name: record(name)
        for name in ("first", "second", "third", "fourth", "deferred")
    }
    # Execute only these fixed inert fixtures, never repository/analyzed source.
    exec(compile(source, "<trusted-default-fixture>", "exec"), namespace)
    assert events == ["first", "second", "third", "fourth"]
    assert _call_names(_flow(source, owner)) == tuple(events)


def test_nested_lambda_default_evaluates_only_current_creation_defaults() -> None:
    source = (
        "stored = lambda value=(lambda inner=first(): deferred_inner()), "
        "*, other=second(): deferred_outer()"
    )
    assert _call_names(_flow(source)) == ("first", "second")


def test_default_nested_in_deferred_lambda_body_stays_deferred() -> None:
    source = (
        "stored = lambda value=first(): "
        "(lambda inner=deferred_default(): deferred_body())"
    )
    assert _call_names(_flow(source)) == ("first",)
    dependencies = ModuleLexicalDependencyProjection.from_module(ast.parse(source))
    phases = {
        surface.reference.id: surface.binding_phase
        for surface in dependencies.name_surfaces
    }
    assert phases["first"] is ModuleBindingResolutionPhase.SOURCE_POSITION
    assert phases["deferred_default"] is ModuleBindingResolutionPhase.FINAL_MODULE
    assert phases["deferred_body"] is ModuleBindingResolutionPhase.FINAL_MODULE


def test_lambda_body_dependencies_remain_with_actual_source_identity_and_scope() -> (
    None
):
    module = ast.parse("stored = lambda value=external_default(): external_body(value)")
    declaration = module.body[0]
    expression = declaration.value
    assert isinstance(expression, ast.Lambda)
    default_call = expression.args.defaults[0]
    body_call = expression.body
    dependencies = ModuleLexicalDependencyProjection.from_module(module)
    (default_surface,) = tuple(
        surface
        for surface in dependencies.name_surfaces
        if surface.reference is default_call.func
    )
    (body_surface,) = tuple(
        surface
        for surface in dependencies.name_surfaces
        if surface.reference is body_call.func
    )
    assert default_surface.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
    assert body_surface.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
    assert default_surface.use is DeclarationDependencyUse.EXECUTION
    assert body_surface.use is DeclarationDependencyUse.EXECUTION
    assert {surface.reference.id for surface in dependencies.name_surfaces} == {
        "external_default",
        "external_body",
    }


def test_eager_name_loads_keep_default_but_exclude_lambda_body() -> None:
    module = ast.parse("stored = lambda value=Target(): Target(value)")
    expression = module.body[0].value
    assert isinstance(expression, ast.Lambda)
    references = EagerNameLoadCollector.collect(module, "Target")
    assert len(references) == 1
    assert references[0] is expression.args.defaults[0].func


def test_shared_default_visitor_preserves_actual_expression_identity_and_order() -> (
    None
):
    function = ast.parse(
        "def sample(a=first(), /, b=second(), *items, required, "
        "c=third(), d=None, **options): pass"
    ).body[0]

    class RecordDefaults(FunctionDefaultVisitor):
        def __init__(self):
            self.expressions = []

        def visit(self, node):
            self.expressions.append(node)

    collector = RecordDefaults()
    collector.visit_argument_defaults(function.args)
    expected = (*function.args.defaults, *function.args.kw_defaults[1:])
    assert len(collector.expressions) == len(expected)
    assert all(
        actual is original
        for actual, original in zip(collector.expressions, expected, strict=True)
    )


def test_shared_default_visitor_lambda_never_visits_body() -> None:
    expression = ast.parse(
        "lambda value=first(), *, other=second(): deferred()", mode="eval"
    ).body

    class RecordDefaults(FunctionDefaultVisitor):
        def __init__(self):
            self.calls = []

        def visit_Call(self, node):
            self.calls.append(node)

    collector = RecordDefaults()
    collector.visit(expression)
    expected = (*expression.args.defaults, *expression.args.kw_defaults)
    assert len(collector.calls) == len(expected)
    assert all(
        actual is original
        for actual, original in zip(collector.calls, expected, strict=True)
    )


@pytest.mark.parametrize(
    "source",
    (
        "def f(value=1): return value",
        "def f(*, value): return value",
        "def f(*, value=1): return value",
        "def f(value, other=2): return value",
        "def f(value: int = None): return value",
        "def f(*, value: int = None, other=2): return value",
        "def f(*values, value=1, **options): return value",
    ),
)
def test_parameter_removal_dependency_projection_keeps_default_expressions(
    source: str,
) -> None:
    function = ast.parse(source).body[0]
    expected = function.body[0].value
    (reference,) = FunctionParameterBinding(function, "value").required_references()
    assert reference is expected
