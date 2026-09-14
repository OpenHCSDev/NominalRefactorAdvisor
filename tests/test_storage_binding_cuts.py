"""Storage-before-write and lexical outer lookup are different obligations."""

import ast
import builtins
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    InitialNativeFrame,
    InitialNativeIsland,
    OpenCapturedReference,
)
from nominal_refactor_advisor.product_flow import (
    InitialCompactParameterBinding,
    OpenCompactBindingMutation,
    source_product_flow_projection,
)


def context(source, qualname=""):
    module = ParsedModule(Path("cuts.py"), "cuts", False, ast.parse(source), source)
    return next(
        item
        for item in source_product_flow_projection(module).compact.flow_contexts
        if item.flow.owner.qualname == qualname
    )


def frame():
    local = {}
    global_ = {}
    island = InitialNativeIsland((builtins,), (local, global_))
    return InitialNativeFrame(
        island.namespace_for_storage(local),
        island.namespace_for_storage(global_),
        island.namespace_for_storage(vars(builtins)),
    )


def test_first_write_cut_proves_empty_storage_without_changing_lexical_selection():
    flow = context("property = object\n").flow
    write = flow.mutations[0]
    assert flow.stored_binding_resolution_for("property", write.position) is None
    assert isinstance(
        flow.binding_resolution_for("property", write.position),
        OpenCompactBindingMutation,
    )


def test_repeated_same_site_write_is_not_assumed_first_iteration():
    flow = context("for item in items:\n    property = object\n").flow
    write = next(
        item for item in flow.mutations if item.target.bound_name == "property"
    )
    assert isinstance(
        flow.stored_binding_resolution_for("property", write.position),
        OpenCompactBindingMutation,
    )


def test_conditional_prior_write_stays_open_at_later_overwrite_cut():
    flow = context(
        "if condition:\n    property = object\nproperty = replacement\n"
    ).flow
    write = flow.mutations[-1]
    assert isinstance(
        flow.stored_binding_resolution_for("property", write.position),
        OpenCompactBindingMutation,
    )


def test_future_function_local_storage_can_be_empty_without_global_fallback():
    scope = context("def run():\n    seen = property\n    property = object\n", "run")
    use = scope.flow.callable_reference_uses[0]
    assert scope.flow.stored_binding_resolution_for("property", use.position) is None
    actual = frame()
    assert actual.initial_lookup_namespaces(scope, "property") == (actual.locals,)


def test_parameter_entry_remains_explicit_and_hides_outer_lookup():
    scope = context("def run(property):\n    seen = property\n", "run")
    use = scope.flow.callable_reference_uses[0]
    assert isinstance(
        scope.flow.stored_binding_resolution_for("property", use.position),
        InitialCompactParameterBinding,
    )
    actual = frame()
    assert actual.initial_lookup_namespaces(scope, "property") == (actual.locals,)


def test_declared_global_does_not_become_a_function_local():
    scope = context(
        "def run():\n    global property\n    seen = property\n    property = object\n",
        "run",
    )
    actual = frame()
    assert actual.initial_lookup_namespaces(scope, "property") == (
        actual.globals,
        actual.builtins,
    )


def test_nonlocal_requires_its_missing_closure_relation():
    scope = context(
        "def outer():\n    property = object\n    def run():\n        nonlocal property\n        seen = property\n",
        "outer.run",
    )
    assert isinstance(
        frame().initial_lookup_namespaces(scope, "property"), OpenCapturedReference
    )
