"""Resolve the names in class bases and annotations to the scanned classes they mean."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from .ast_projection import ExpressionReference
from .ast_tools import ParsedModule


def iter_class_definitions(
    statements: list[ast.stmt],
    *,
    parent_qualname: str | None = None,
) -> tuple[tuple[str, ast.ClassDef], ...]:
    classes: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if not isinstance(statement, ast.ClassDef):
            continue
        qualname = (
            statement.name
            if parent_qualname is None
            else f"{parent_qualname}.{statement.name}"
        )
        classes.append((qualname, statement))
        classes.extend(
            iter_class_definitions(list(statement.body), parent_qualname=qualname)
        )
    return tuple(classes)


class _UniqueKnownSymbolSuffixIndex:
    """Resolve only requested suffixes from terminal-name candidate buckets."""

    __slots__ = ("_matches_by_suffix", "_symbols_by_terminal_name")

    def __init__(self, known_symbols: frozenset[str]) -> None:
        symbols_by_terminal_name: dict[str, list[str]] = defaultdict(list)
        for symbol in known_symbols:
            symbols_by_terminal_name[symbol.rsplit(".", 1)[-1]].append(symbol)
        self._symbols_by_terminal_name = {
            terminal_name: tuple(symbols)
            for terminal_name, symbols in symbols_by_terminal_name.items()
        }
        self._matches_by_suffix: dict[str, str | None] = {}

    @classmethod
    def from_terminal_buckets(
        cls,
        symbols_by_terminal_name: dict[str, tuple[str, ...]],
    ) -> _UniqueKnownSymbolSuffixIndex:
        index = cls.__new__(cls)
        index._symbols_by_terminal_name = symbols_by_terminal_name
        index._matches_by_suffix = {}
        return index

    def get(self, suffix: str) -> str | None:
        if suffix in self._matches_by_suffix:
            return self._matches_by_suffix[suffix]
        qualified_suffix = f".{suffix}"
        match: str | None = None
        for symbol in self._symbols_by_terminal_name.get(suffix.rsplit(".", 1)[-1], ()):
            if symbol != suffix and not symbol.endswith(qualified_suffix):
                continue
            if match is not None and match != symbol:
                match = None
                break
            match = symbol
        self._matches_by_suffix[suffix] = match
        return match

    def root_relative_match(self, qualified_name: str) -> str | None:
        """Resolve one imported name across a narrower analysis-root boundary."""

        matches = tuple(
            symbol
            for symbol in self._symbols_by_terminal_name.get(
                qualified_name.rsplit(".", 1)[-1], ()
            )
            if symbol == qualified_name
            or qualified_name.endswith(f".{symbol}")
            or symbol.endswith(f".{qualified_name}")
        )
        return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=8)
def _unique_known_symbol_by_suffix(
    known_symbols: frozenset[str],
) -> _UniqueKnownSymbolSuffixIndex:
    """Share a bounded lazy suffix resolver for one repository symbol set."""

    return _UniqueKnownSymbolSuffixIndex(known_symbols)


def _referenced_bases(
    node: ast.ClassDef,
) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
    """Each base a name or attribute chain spells, with that chain; `Family[Key]` spells `Family`."""

    for base in node.bases:
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(base)
        ).attribute_chain
        if parts is not None:
            yield base, parts


def _terminal_reference_name(node: ast.AST) -> str | None:
    parts = ExpressionReference.decoded(
        ClassSymbolResolutionAuthority.reference_node(node)
    ).attribute_chain
    return None if parts is None else parts[-1]


@dataclass(frozen=True)
class ClassSymbolResolutionAuthority:
    """Resolve AST name chains to indexed class symbols under an explicit policy."""

    NOMINAL_FAMILY_NEUTRAL_BASE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"ABC", "Generic", "Protocol", "object"}
    )

    parsed_module: ParsedModule
    import_aliases: dict[str, str]
    known_symbols: frozenset[str]
    unique_symbols_by_name: dict[str, str]
    allow_unique_unqualified: bool

    def symbol_for_node(self, node: ast.AST) -> str | None:
        parts = ExpressionReference.decoded(self.reference_node(node)).attribute_chain
        if parts is None:
            return None
        alias_symbol = self._import_alias_symbol(parts)
        if alias_symbol is not None:
            return alias_symbol
        module_local_symbol = self._module_local_symbol(parts)
        if module_local_symbol is not None:
            return module_local_symbol
        if self.allow_unique_unqualified:
            return self._unique_unqualified_symbol(parts)
        return None

    @staticmethod
    def reference_node(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Subscript):
            return node.value
        return node

    def _import_alias_symbol(self, parts: tuple[str, ...]) -> str | None:
        first, *rest = parts
        alias_target = self.import_aliases.get(first)
        if alias_target is None:
            return None
        candidate = ".".join((alias_target, *rest)) if rest else alias_target
        if candidate in self.known_symbols:
            return candidate
        # A scan may start at a package subdirectory, making indexed symbols
        # source-root-relative while imports retain their full package prefix.
        # Resolve only a unique suffix match so unrelated same-named classes do
        # not create a speculative inheritance edge.
        unique_symbol_by_suffix = _unique_known_symbol_by_suffix(self.known_symbols)
        return unique_symbol_by_suffix.root_relative_match(candidate)

    def _module_local_symbol(self, parts: tuple[str, ...]) -> str | None:
        candidate = ".".join((self.parsed_module.module_name, *parts))
        if candidate in self.known_symbols:
            return candidate
        return None

    def _unique_unqualified_symbol(self, parts: tuple[str, ...]) -> str | None:
        if len(parts) != 1:
            return None
        return self.unique_symbols_by_name.get(parts[0])

    @classmethod
    def declared_base_name(cls, node: ast.AST) -> str | None:
        reference_node = cls.reference_node(node)
        if ExpressionReference.decoded(reference_node).attribute_chain is None:
            return None
        return ast.unparse(reference_node)

    @classmethod
    def establishes_nominal_family(cls, declared_base_name: str) -> bool:
        """Return whether a base declaration can own a domain class family."""

        return (
            declared_base_name.rsplit(".", 1)[-1]
            not in cls.NOMINAL_FAMILY_NEUTRAL_BASE_NAMES
        )
