"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .ast_tools import ParsedModule


@dataclass(frozen=True)
class IndexedClass:
    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    node: ast.ClassDef
    declared_base_names: tuple[str, ...]
    resolved_base_symbols: tuple[str, ...]


@dataclass(frozen=True)
class ClassFamilyIndex:
    classes_by_symbol: dict[str, IndexedClass]
    symbols_by_simple_name: dict[str, tuple[str, ...]]
    symbols_by_file_and_qualname: dict[tuple[str, str], str]
    children_by_symbol: dict[str, tuple[str, ...]]
    descendants_by_symbol: dict[str, tuple[str, ...]]

    def class_for(self, symbol: str) -> IndexedClass | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        return self.descendants_by_symbol.get(base_symbol, ())


def _iter_class_defs(
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
            _iter_class_defs(list(statement.body), parent_qualname=qualname)
        )
    return tuple(classes)


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _resolve_relative_module(
    parsed_module: ParsedModule,
    *,
    imported_module: str | None,
    level: int,
) -> str | None:
    if level == 0:
        return imported_module
    package_parts = parsed_module.module_name.split(".")
    if not parsed_module.is_package_init:
        package_parts = package_parts[:-1]
    if level > 1:
        if level - 1 > len(package_parts):
            return None
        package_parts = package_parts[: len(package_parts) - (level - 1)]
    if imported_module:
        return ".".join((*package_parts, *imported_module.split(".")))
    return ".".join(package_parts)


@lru_cache(maxsize=None)
def _module_import_aliases(parsed_module: ParsedModule) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in parsed_module.module.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                aliases[local_name] = (
                    alias.name if alias.asname else alias.name.split(".", 1)[0]
                )
        elif isinstance(statement, ast.ImportFrom):
            resolved_module = _resolve_relative_module(
                parsed_module,
                imported_module=statement.module,
                level=statement.level,
            )
            if resolved_module is None:
                continue
            for alias in statement.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                aliases[local_name] = f"{resolved_module}.{alias.name}"
    return aliases


def _resolve_base_symbol(
    parsed_module: ParsedModule,
    *,
    base_node: ast.AST,
    import_aliases: dict[str, str],
    known_symbols: frozenset[str],
    unique_symbols_by_name: dict[str, str],
) -> str | None:
    parts = _attribute_chain(base_node)
    if parts is None:
        return None
    first, *rest = parts
    alias_target = import_aliases.get(first)
    if alias_target is not None:
        candidate = ".".join((alias_target, *rest)) if rest else alias_target
        if candidate in known_symbols:
            return candidate
    module_local_candidate = ".".join((parsed_module.module_name, *parts))
    if module_local_candidate in known_symbols:
        return module_local_candidate
    if len(parts) == 1:
        return unique_symbols_by_name.get(first)
    return None


def build_class_family_index(modules: list[ParsedModule]) -> ClassFamilyIndex:
    return _build_class_family_index_cached(tuple(modules))


@lru_cache(maxsize=None)
def _build_class_family_index_cached(
    modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    raw_classes: list[tuple[ParsedModule, str, ast.ClassDef]] = []
    for parsed_module in modules:
        for qualname, node in _iter_class_defs(list(parsed_module.module.body)):
            raw_classes.append((parsed_module, qualname, node))

    class_records = [
        (
            parsed_module,
            qualname,
            node,
            f"{parsed_module.module_name}.{qualname}",
        )
        for parsed_module, qualname, node in raw_classes
    ]
    known_symbols = frozenset(symbol for _, _, _, symbol in class_records)
    symbols_by_simple_name_multimap: dict[str, list[str]] = defaultdict(list)
    for _, qualname, _, symbol in class_records:
        symbols_by_simple_name_multimap[qualname.rsplit(".", 1)[-1]].append(symbol)
    unique_symbols_by_name = {
        name: symbols[0]
        for name, symbols in symbols_by_simple_name_multimap.items()
        if len(symbols) == 1
    }

    classes_by_symbol: dict[str, IndexedClass] = {}
    symbols_by_file_and_qualname: dict[tuple[str, str], str] = {}
    children_by_symbol_lists: dict[str, list[str]] = defaultdict(list)

    for parsed_module, qualname, node, symbol in class_records:
        import_aliases = _module_import_aliases(parsed_module)
        resolved_base_symbols = tuple(
            resolved
            for base in node.bases
            if (
                resolved := _resolve_base_symbol(
                    parsed_module,
                    base_node=base,
                    import_aliases=import_aliases,
                    known_symbols=known_symbols,
                    unique_symbols_by_name=unique_symbols_by_name,
                )
            )
            is not None
        )
        indexed_class = IndexedClass(
            symbol=symbol,
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=str(parsed_module.path),
            line=node.lineno,
            node=node,
            declared_base_names=tuple(
                ast.unparse(base)
                for base in node.bases
                if _attribute_chain(base) is not None
            ),
            resolved_base_symbols=resolved_base_symbols,
        )
        classes_by_symbol[symbol] = indexed_class
        symbols_by_file_and_qualname[(str(parsed_module.path), qualname)] = symbol
        for base_symbol in resolved_base_symbols:
            children_by_symbol_lists[base_symbol].append(symbol)

    symbols_by_simple_name = {
        name: tuple(sorted(symbols))
        for name, symbols in symbols_by_simple_name_multimap.items()
    }
    children_by_symbol = {
        symbol: tuple(sorted(children))
        for symbol, children in children_by_symbol_lists.items()
    }
    descendants_by_symbol: dict[str, tuple[str, ...]] = {}
    for symbol in sorted(classes_by_symbol):
        descendants: list[str] = []
        queue = list(children_by_symbol.get(symbol, ()))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            descendants.append(current)
            queue.extend(children_by_symbol.get(current, ()))
        if descendants:
            descendants_by_symbol[symbol] = tuple(descendants)
    return ClassFamilyIndex(
        classes_by_symbol=classes_by_symbol,
        symbols_by_simple_name=symbols_by_simple_name,
        symbols_by_file_and_qualname=symbols_by_file_and_qualname,
        children_by_symbol=children_by_symbol,
        descendants_by_symbol=descendants_by_symbol,
    )
