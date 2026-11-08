"""Declaration enumeration reuses the index without changing scope or header answers."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from nominal_refactor_advisor.ast_tools import ModuleSyntaxIndex, ParsedModule, parse_python_modules
from nominal_refactor_advisor.relocation_observations import (
    enclosing_symbols,
    scoped_classes,
    scoped_definitions,
    walk_scopes,
)


@pytest.fixture
def scoped_module(tmp_path: Path) -> ParsedModule:
    (tmp_path / "scopes.py").write_text(
        "@decorate(mark)\n"
        "class Outer(base(mark)):\n"
        "    @decorate(mark)\n"
        "    def method(self, arg: mark = mark) -> mark:\n"
        "        if condition:\n"
        "            class Inner:\n"
        "                def local(self): return mark\n"
        "        def nested(): return mark\n"
        "if condition:\n"
        "    class Conditional: pass\n"
        "async def final(arg=mark): return mark\n"
        "class Twin: pass\n"
        "class Twin: pass\n",
        encoding="utf-8",
    )
    (module,) = parse_python_modules(tmp_path, use_parse_cache=False)
    return module


def test_scope_enumeration_preserves_every_declaration_and_legacy_order_without_rewalking(
    scoped_module: ParsedModule,
) -> None:
    index = scoped_module.syntax_index
    original_classes = index.indexed_nodes_of_type(ast.ClassDef)
    with patch("ast.iter_child_nodes", wraps=ast.iter_child_nodes) as child_walks:
        classes = tuple(scoped_classes(scoped_module))
        definitions = tuple(scoped_definitions(scoped_module))
        scopes = tuple(walk_scopes(scoped_module))

    assert [(name, node.lineno) for name, node in classes] == [
        ("Outer", 2), ("Conditional", 10), ("Twin", 12), ("Twin", 13),
        ("Outer.method.Inner", 6),
    ]
    assert {id(node) for _name, node in classes} == {
        id(node) for _index, node in original_classes
    }
    owners = {
        node.lineno: node
        for node in index.depth_first_nodes
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert [(item.qualname, id(item.owner)) for item in definitions] == [
        ("final", id(scoped_module.module)), ("Outer.method", id(owners[2])),
        ("Outer.method.nested", id(owners[4])), ("Outer.method.Inner.local", id(owners[6])),
    ]
    assert [prefix for prefix, _scope in scopes] == [
        "", "Twin.", "Twin.", "final.", "Conditional.", "Outer.",
        "Outer.method.", "Outer.method.nested.", "Outer.method.Inner.",
        "Outer.method.Inner.local.",
    ]
    # The parse's existing index already visited every node and recorded lexical owners.
    assert child_walks.call_count == 0


def test_cold_scope_consumers_build_the_shared_index_once(scoped_module: ParsedModule) -> None:
    with patch.object(ModuleSyntaxIndex, "build", wraps=ModuleSyntaxIndex.build) as builds:
        tuple(scoped_classes(scoped_module))
        tuple(scoped_definitions(scoped_module))
        tuple(walk_scopes(scoped_module))
    assert builds.call_count == 1


def test_definition_headers_keep_their_enclosing_owner_not_the_definition_body(
    scoped_module: ParsedModule,
) -> None:
    symbols = enclosing_symbols(scoped_module)
    mark_owners = {
        (node.lineno, node.col_offset): symbols[id(node)]
        for _index, node in scoped_module.syntax_index.indexed_nodes_of_type(ast.Name)
        if node.id == "mark"
    }
    assert mark_owners == {
        (1, 10): "<module>",
        (2, 17): "<module>",
        (3, 14): "Outer",
        (4, 26): "Outer",
        (4, 33): "Outer",
        (4, 42): "Outer",
        (7, 40): "Outer.method.Inner.local",
        (8, 29): "Outer.method.nested",
        (11, 20): "<module>",
        (11, 34): "final",
    }
