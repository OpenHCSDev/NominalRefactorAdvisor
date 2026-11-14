"""The syntax index answers every declaration and who evaluates each node, without rewalking."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from nominal_refactor_advisor.ast_tools import ModuleSyntaxIndex, ParsedModule, parse_python_modules
from nominal_refactor_advisor.relocation_observations import ModuleClassReadings, ModuleDefinitions


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
        "async def final(arg=mark): return (lambda: mark)\n"
        "class Twin: pass\n"
        "class Twin: pass\n",
        encoding="utf-8",
    )
    (module,) = parse_python_modules(tmp_path, use_parse_cache=False)
    return module


def test_declarations_name_every_class_and_def_in_source_order_without_rewalking(
    scoped_module: ParsedModule,
) -> None:
    index = scoped_module.syntax_index
    original_classes = index.indexed_nodes_of_type(ast.ClassDef)
    with patch("ast.iter_child_nodes", wraps=ast.iter_child_nodes) as child_walks:
        classes = index.declarations(ast.ClassDef)
        definitions = index.declarations((ast.FunctionDef, ast.AsyncFunctionDef))

    assert [(name, node.lineno) for name, node in classes] == [
        ("Outer", 2), ("Outer.method.Inner", 6), ("Conditional", 10), ("Twin", 12),
        ("Twin", 13),
    ]
    assert {id(node) for _name, node in classes} == {
        id(node) for _index, node in original_classes
    }
    assert [(name, node.lineno) for name, node in definitions] == [
        ("Outer.method", 4), ("Outer.method.Inner.local", 7), ("Outer.method.nested", 8),
        ("final", 11),
    ]
    # The parse's existing index already visited every node and recorded lexical owners.
    assert child_walks.call_count == 0


def test_cold_declaration_facts_build_the_shared_index_once(scoped_module: ParsedModule) -> None:
    with patch.object(ModuleSyntaxIndex, "build", wraps=ModuleSyntaxIndex.build) as builds:
        classes = scoped_module.fact(ModuleClassReadings).classes
        definitions = scoped_module.fact(ModuleDefinitions).definitions
    assert builds.call_count == 1
    assert [reading.qualname for reading in classes] == [
        "Outer", "Outer.method.Inner", "Conditional", "Twin", "Twin",
    ]
    assert [definition.qualname for definition in definitions] == [
        "Outer.method", "Outer.method.Inner.local", "Outer.method.nested", "final",
    ]


def test_definition_headers_are_evaluated_by_their_enclosing_owner_not_the_definition_body(
    scoped_module: ParsedModule,
) -> None:
    index = scoped_module.syntax_index
    mark_owners = {
        (node.lineno, node.col_offset): index.evaluating_symbol(node_index)
        for node_index, node in index.indexed_nodes_of_type(ast.Name)
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
        # A lambda is not a scope: the def around it evaluates its body.
        (11, 43): "final",
    }
