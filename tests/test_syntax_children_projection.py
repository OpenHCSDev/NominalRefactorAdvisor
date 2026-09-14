"""Completion uses original indexed children, not another live AST traversal."""

import ast
from copy import deepcopy

import pytest

from nominal_refactor_advisor.ast_tools import module_syntax_index
from test_source_body_completion import entry_for


def test_children_preserve_original_order_and_share_parent_authority(monkeypatch):
    module = ast.parse("class Parent(Base):\n    first = 1\n    class Child: pass\n")
    owner = module.body[0]
    syntax = module_syntax_index(module)
    expected = tuple(ast.iter_child_nodes(owner))

    def forbidden(*args):
        raise AssertionError("Indexed projection must not traverse the AST again")

    monkeypatch.setattr(ast, "iter_child_nodes", forbidden)
    children = syntax.children_by_node
    assert children[owner] == expected
    assert children is syntax.children_by_node
    assert all(
        syntax.parent_by_node[child] is parent
        for parent, nodes in children.items()
        for child in nodes
    )


@pytest.mark.parametrize("same_parent", (False, True))
def test_ambiguous_original_child_cannot_gain_an_owner(same_parent):
    module = ast.parse("class First: pass\nclass Second: pass\n")
    first, second = module.body
    shared = first.body[0]
    (first if same_parent else second).body.append(shared)
    syntax = module_syntax_index(module)
    assert shared not in syntax.parent_by_node
    assert all(shared not in nodes for nodes in syntax.children_by_node.values())


def test_lazily_derived_children_still_refer_to_original_tree():
    module = ast.parse("class First:\n    value = 7\n    pass\n")
    owner = module.body[0]
    originals = tuple(owner.body)
    syntax = module_syntax_index(module)
    owner.body[:] = [deepcopy(originals[0])]
    assert syntax.children_by_node[owner] == originals


def test_completion_does_not_repeat_whole_module_syntax_walk():
    entry = entry_for("key = None\npass")
    syntax = module_syntax_index(entry.execution.module.module)
    expected = entry.final_evaluation

    class NoTraversal(tuple):
        def __iter__(self):
            raise AssertionError("Completion should query this owner's children")

    original = syntax.depth_first_nodes
    object.__setattr__(syntax, "depth_first_nodes", NoTraversal(original))
    try:
        assert entry.final_evaluation is expected
        entry.node.body.pop()
        with pytest.raises(ValueError, match="original body statements"):
            _ = entry.final_evaluation
    finally:
        object.__setattr__(syntax, "depth_first_nodes", original)
