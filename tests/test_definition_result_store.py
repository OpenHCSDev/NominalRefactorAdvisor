"""One source definition owns decorator correspondence and final native storage."""

import ast
from copy import copy

import pytest

from nominal_refactor_advisor.source_execution import SourceCreatedFunctionCapture
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_function_result import execution


def definition(source):
    environment = execution(source)
    node = environment.module.module.body[-1]
    entry = (
        environment.class_entry(node)
        if isinstance(node, ast.ClassDef)
        else SourceCreatedFunctionCapture(environment, node)
    )
    return environment, entry


@pytest.mark.parametrize("kind", ("class", "function"))
@pytest.mark.parametrize("decorated", (False, True))
def test_source_definition_selects_the_original_final_store(kind, decorated):
    source = "@staticmethod\n" if decorated else ""
    source += "class Target: pass\n" if kind == "class" else "def Target(): pass\n"
    environment, entry = definition(source)
    store = entry.native_result_store
    node = entry.node
    production = node.decorator_list[0] if decorated else node
    assert store is environment.module.native_compilation.value_store_for(
        SourceByteSpan.require_node(production),
        SourceByteSpan.require_node(node),
        "Target",
    )
    assert entry.decorator_nodes == tuple(node.decorator_list)
    assert store.frame.resolve(entry) is entry.parent_context
    if kind == "class":
        if decorated:
            with pytest.raises(
                ValueError, match="Class decorator result remains unproved"
            ):
                entry.result()
        else:
            assert entry.result().production is store
    elif decorated:
        assert entry.result().production is store


@pytest.mark.parametrize("kind", ("class", "function"))
@pytest.mark.parametrize("damage", ("copy", "reverse", "omit", "duplicate"))
def test_warmed_store_selection_revalidates_original_decorator_occurrences(
    kind, damage
):
    source = "@classmethod\n@staticmethod\n"
    source += "class Target: pass\n" if kind == "class" else "def Target(): pass\n"
    _, entry = definition(source)
    entry.native_result_store
    nodes = entry.node.decorator_list
    if damage == "copy":
        nodes[0] = copy(nodes[0])
    elif damage == "reverse":
        nodes.reverse()
    elif damage == "omit":
        nodes.pop()
    else:
        nodes[0] = nodes[1]
    with pytest.raises(ValueError, match="original source chain"):
        entry.native_result_store
