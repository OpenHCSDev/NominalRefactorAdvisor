from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor import class_index as class_index_module
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    collect_family_items,
    parse_python_modules,
)
from nominal_refactor_advisor.class_index import (
    ClassSymbolResolutionAuthority,
    CompactClassReferenceResolver,
    CompactModuleClassProjectionFamily,
    build_class_family_index,
    build_compact_class_family_index,
)


def _resolution(
    known_symbols: frozenset[str],
) -> ClassSymbolResolutionAuthority:
    return ClassSymbolResolutionAuthority(
        parsed_module=ParsedModule(
            path=Path("pkg/consumer.py"),
            module_name="consumer",
            is_package_init=False,
            module=ast.parse(""),
            source="",
        ),
        import_aliases={"external": "full.pkg.types"},
        known_symbols=known_symbols,
        unique_symbols_by_name={},
        allow_unique_unqualified=False,
    )


@pytest.mark.parametrize(
    ("source", "expected_safe"),
    (
        (
            "from typing import final\n"
            "@final\n"
            "class Target:\n"
            "    pass\n"
            "final = replacement\n",
            True,
        ),
        (
            "from typing import final\n"
            "final = replacement\n"
            "@final\n"
            "class Target:\n"
            "    pass\n",
            False,
        ),
    ),
)
def test_method_promotion_decorator_safety_uses_declaration_time_bindings(
    source: str,
    expected_safe: bool,
) -> None:
    module = ParsedModule(
        path=Path("pkg/decorators.py"),
        module_name="pkg.decorators",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )

    indexed_class = build_class_family_index([module]).classes_by_symbol[
        "pkg.decorators.Target"
    ]

    assert indexed_class.class_decorators_are_promotion_safe is expected_safe


def test_import_alias_suffix_index_preserves_unique_root_relative_match() -> None:
    resolution = _resolution(
        frozenset(
            {
                "types.Target",
                "other.Unrelated",
            }
        )
    )

    assert resolution.symbol_for_node(ast.parse("external.Target").body[0].value) == (
        "types.Target"
    )


def test_import_alias_suffix_index_preserves_ambiguous_fail_closed_result() -> None:
    resolution = _resolution(
        frozenset(
            {
                "left.types.Target",
                "right.types.Target",
            }
        )
    )

    assert (
        resolution.symbol_for_node(ast.parse("external.Target").body[0].value) is None
    )


def test_import_alias_suffix_index_is_lazy_and_repository_bounded() -> None:
    symbols = frozenset(
        {
            *(f"root.package_{index}.types.Target_{index}" for index in range(100)),
            "root.unique.types.Unique",
            "root.left.types.Target",
            "root.right.types.Target",
        }
    )
    class_index_module._unique_known_symbol_by_suffix.cache_clear()

    suffix_index = class_index_module._unique_known_symbol_by_suffix(symbols)

    assert suffix_index._matches_by_suffix == {}
    assert sum(
        len(bucket) for bucket in suffix_index._symbols_by_terminal_name.values()
    ) == len(symbols)
    assert suffix_index.get("types.Unique") == "root.unique.types.Unique"
    assert suffix_index.get("types.Target") is None
    assert set(suffix_index._matches_by_suffix) == {"types.Unique", "types.Target"}
    assert class_index_module._unique_known_symbol_by_suffix.cache_parameters() == {
        "maxsize": 8,
        "typed": False,
    }


def test_compact_class_family_index_matches_full_ast_inheritance_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "base.py").write_text(
        "class Root:\n    pass\n\nclass Mid(Root):\n    pass\n",
        encoding="utf-8",
    )
    (package_root / "leaf.py").write_text(
        "from pkg.base import Mid as ImportedMid\n"
        "\n"
        "class Leaf(ImportedMid):\n"
        "    pass\n"
        "\n"
        "class GenericLeaf(ImportedMid[int]):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "qualified.py").write_text(
        "import pkg.base as base_alias\n"
        "\n"
        "class Qualified(base_alias.Root):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "unique.py").write_text(
        "class UniqueLeaf(Root):\n    pass\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    full_index = build_class_family_index(modules)
    original_walk_nodes = class_index_module._walk_nodes
    walked_roots: list[ast.AST] = []

    def tracked_walk_nodes(root: ast.AST) -> tuple[ast.AST, ...]:
        walked_roots.append(root)
        return original_walk_nodes(root)

    monkeypatch.setattr(class_index_module, "_walk_nodes", tracked_walk_nodes)
    compact_projections = tuple(
        projection
        for module in modules
        for projection in collect_family_items(
            module,
            CompactModuleClassProjectionFamily,
        )
    )
    compact_index = build_compact_class_family_index(compact_projections)
    compact_resolver = CompactClassReferenceResolver.from_index(
        compact_projections,
        compact_index,
    )

    assert walked_roots == [
        module.module for module in modules for _root_walk in range(4)
    ]
    assert compact_resolver.known_symbols is compact_index.classes_by_symbol
    assert (
        compact_resolver.unique_symbols_by_suffix._symbols_by_terminal_name
        is compact_index.symbols_by_simple_name
    )
    assert set(compact_index.classes_by_symbol) == set(full_index.classes_by_symbol)
    for symbol, full_class in full_index.classes_by_symbol.items():
        compact_class = compact_index.classes_by_symbol[symbol]
        assert compact_class.declared_base_names == full_class.declared_base_names
        assert compact_class.resolved_base_symbols == full_class.resolved_base_symbols
    assert compact_index.symbols_by_simple_name == full_index.symbols_by_simple_name
    assert compact_index.children_by_symbol == full_index.children_by_symbol
    assert compact_index.ancestors_by_symbol == full_index.ancestors_by_symbol
    assert compact_index.descendants_by_symbol == full_index.descendants_by_symbol
