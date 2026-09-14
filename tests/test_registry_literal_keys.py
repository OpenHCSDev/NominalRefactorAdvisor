"""Literal registry keys use native mapping equality through the shared owner."""

import ast

import pytest

from nominal_refactor_advisor.manual_registry import (
    AutoRegisterInstanceViewComponent,
    DirectManualRegistryComponent,
    RegistryEntries,
    DirectModuleClassGraph,
)
from nominal_refactor_advisor.value_expression import LiteralExpressionEffects
from test_registry_key_equivalence import _MANUAL_SOURCE, _parsed


@pytest.mark.parametrize(
    "left,right,distinct",
    (
        ("1", "True", False),
        ("1", "1.0", False),
        ("1", "1+0j", False),
        ("-0.0", "0.0", False),
        ("(1, 'same')", "(True, 'same')", False),
        ("'alpha'", "'beta'", True),
        ("b'alpha'", "'alpha'", True),
        ("None", "()", True),
    ),
)
def test_original_literal_keys_follow_native_dictionary_equivalence(
    left, right, distinct
):
    source = _MANUAL_SOURCE.replace("Mode.ALPHA", left).replace("Mode.BETA", right)
    parsed = _parsed(source)
    namespace = {}
    exec(compile(source, "<literal-key-control>", "exec"), namespace)
    assert len(namespace["REGISTRY"]) == (2 if distinct else 1)
    if distinct:
        component = DirectManualRegistryComponent.from_module_anchor(
            parsed.module, "AlphaHandler"
        )
        assert len(component.entries) == 2
    else:
        with pytest.raises(ValueError, match="keys must be unique"):
            DirectManualRegistryComponent.from_module_anchor(
                parsed.module, "AlphaHandler"
            )


@pytest.mark.parametrize(
    "source", ("name", "Mode.ALPHA", "make_key()", "[]", "{}", "{1}", "(1, [])")
)
def test_unproved_or_unhashable_key_never_becomes_a_syntax_token(source):
    module = ast.parse(f"class Owner: pass\nREGISTRY = {{{source}: Owner}}\n")
    (entry,) = DirectManualRegistryComponent.direct_entries(
        module, DirectModuleClassGraph(module).classes_by_name
    )
    with pytest.raises(ValueError, match="key equivalence remains unproved"):
        _ = entry.key_value


def test_both_conversion_families_inherit_one_entry_invariant():
    for family in (DirectManualRegistryComponent, AutoRegisterInstanceViewComponent):
        assert (
            family.require_ordered_unique_entries
            is RegistryEntries.require_ordered_unique_entries
        )
        assert family.class_names is RegistryEntries.class_names
        assert family.class_nodes is RegistryEntries.class_nodes


def test_literal_owner_retains_original_expression_without_claiming_object_identity():
    node = ast.parse("(1, 'alpha')", mode="eval").body
    literal = LiteralExpressionEffects(node)
    assert literal.node is node
    assert literal.hashable_value == (1, "alpha")
