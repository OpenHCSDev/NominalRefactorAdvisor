from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import ClassSymbolResolutionAuthority


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
