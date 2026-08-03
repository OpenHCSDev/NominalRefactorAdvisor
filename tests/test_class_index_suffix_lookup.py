from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    collect_family_items,
    parse_python_modules,
)
from nominal_refactor_advisor.class_index import (
    ClassSymbolResolutionAuthority,
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


def test_compact_class_family_index_matches_full_ast_inheritance_graph(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "base.py").write_text(
        "class Root:\n" "    pass\n" "\n" "class Mid(Root):\n" "    pass\n",
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
        "class UniqueLeaf(Root):\n" "    pass\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    full_index = build_class_family_index(modules)
    compact_index = build_compact_class_family_index(
        tuple(
            projection
            for module in modules
            for projection in collect_family_items(
                module,
                CompactModuleClassProjectionFamily,
            )
        )
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
