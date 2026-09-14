"""Source coordinates do not authenticate a site's ownership of the module AST."""

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, module_syntax_index
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def projection(source="result = property\n"):
    module = ParsedModule(
        Path("membership.py"), "membership", False, ast.parse(source), source
    )
    return source_product_flow_projection(module)


@pytest.mark.parametrize("site_field", ("operations", "evaluations"))
def test_copied_site_node_cannot_borrow_the_original_event_and_cut(site_field):
    original = projection()
    node = original.module.module.body[0].value
    foreign = deepcopy(node)
    sites = getattr(original, site_field)
    assert any(site.node is node for site in sites)
    with pytest.raises(ValueError, match="actual module AST"):
        replace(
            original,
            **{
                site_field: tuple(
                    replace(site, node=foreign) if site.node is node else site
                    for site in sites
                )
            },
        )


def test_reparsed_module_does_not_reown_previous_flow_sites():
    original = projection()
    reparsed = original.module.with_source(original.module.source)
    assert reparsed.module is not original.module.module
    with pytest.raises(ValueError, match="actual module AST"):
        replace(original, module=reparsed)


def test_syntax_owner_shares_membership_across_projections_and_activations():
    original = projection(
        "class First:\n    value = property\nclass Second:\n    value = staticmethod\n"
    )
    copied_container = replace(original)
    syntax = module_syntax_index(original.module.module)
    assert (
        syntax.node_membership
        is module_syntax_index(copied_container.module.module).node_membership
    )
    assert all(
        site.node in syntax.node_membership
        for site in (*original.operations, *original.evaluations)
    )
    assert "node_membership" not in original.__dict__
    environment = SourceModuleExecution.from_source(original)
    environment.require_class_creation(original.module.module.body[-1])
