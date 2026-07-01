"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path

from .ast_tools import ParsedModule
from .collection_algebra import sorted_tuple


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

    @classmethod
    def from_parsed_class(
        cls,
        parsed_module: ParsedModule,
        qualname: str,
        node: ast.ClassDef,
    ) -> "IndexedClass":
        return cls(
            symbol=f"{parsed_module.module_name}.{qualname}",
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=str(parsed_module.path),
            line=node.lineno,
            node=node,
            declared_base_names=tuple(
                declared_base_name
                for base in node.bases
                if (
                    declared_base_name
                    := ClassSymbolResolutionAuthority.declared_base_name(base)
                )
                is not None
            ),
            resolved_base_symbols=(),
        )

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> "IndexedClass":
        return replace(self, resolved_base_symbols=resolved_base_symbols)


@dataclass(frozen=True)
class ClassFamilyIndex:
    classes_by_symbol: dict[str, IndexedClass]
    symbols_by_simple_name: dict[str, tuple[str, ...]]
    symbols_by_file_and_qualname: dict[tuple[str, str], str]
    children_by_symbol: dict[str, tuple[str, ...]]
    ancestors_by_symbol: dict[str, tuple[str, ...]]
    descendants_by_symbol: dict[str, tuple[str, ...]]

    def class_for(self, symbol: str) -> IndexedClass | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        return self.descendants_by_symbol.get(base_symbol, ())

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.ancestors_by_symbol.get(class_symbol, ())

    def class_records_excluding_files(
        self,
        file_paths: frozenset[str],
    ) -> tuple[IndexedClass, ...]:
        return tuple(
            indexed_class
            for indexed_class in self.classes_by_symbol.values()
            if _resolved_path_text(indexed_class.file_path) not in file_paths
        )


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
        classes.extend(_iter_class_defs(list(statement.body), parent_qualname=qualname))
    return tuple(classes)


@dataclass(frozen=True)
class AttributeChainAuthority:
    def project(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            parent = self.project(node.value)
            if parent is None:
                return None
            return (*parent, node.attr)
        return None


ATTRIBUTE_CHAIN_AUTHORITY = AttributeChainAuthority()


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
                parsed_module, imported_module=statement.module, level=statement.level
            )
            if resolved_module is None:
                continue
            for alias in statement.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                aliases[local_name] = f"{resolved_module}.{alias.name}"
    return aliases


@dataclass(frozen=True)
class ClassSymbolResolutionAuthority:
    """Resolve AST name chains to indexed class symbols under an explicit policy."""

    parsed_module: ParsedModule
    import_aliases: dict[str, str]
    known_symbols: frozenset[str]
    unique_symbols_by_name: dict[str, str]
    allow_unique_unqualified: bool

    def symbol_for_node(self, node: ast.AST) -> str | None:
        parts = ATTRIBUTE_CHAIN_AUTHORITY.project(self.reference_node(node))
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
        return None

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
        if ATTRIBUTE_CHAIN_AUTHORITY.project(reference_node) is None:
            return None
        return ast.unparse(reference_node)


@dataclass(frozen=True)
class ModuleClassReferenceResolver:
    """Resolve class references in expression syntax against a class index."""

    parsed_module: ParsedModule
    class_index: ClassFamilyIndex

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        return frozenset(self.class_index.classes_by_symbol)

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        return {
            simple_name: symbols[0]
            for simple_name, symbols in self.class_index.symbols_by_simple_name.items()
            if len(symbols) == 1
        }

    @cached_property
    def import_aliases(self) -> dict[str, str]:
        return _module_import_aliases(self.parsed_module)

    @cached_property
    def constructor_assignment_symbols(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for statement in self.parsed_module.module.body:
            if not isinstance(statement, ast.Assign | ast.AnnAssign):
                continue
            target_name = _single_assignment_target_name(statement)
            if target_name is None:
                continue
            value = statement.value
            if value is None:
                continue
            symbol = self._direct_constructor_symbol(value)
            if symbol is not None:
                assignments[target_name] = symbol
        return assignments

    @cached_property
    def reference_resolution(self) -> ClassSymbolResolutionAuthority:
        return ClassSymbolResolutionAuthority(
            parsed_module=self.parsed_module,
            import_aliases=self.import_aliases,
            known_symbols=self.known_symbols,
            unique_symbols_by_name=self.unique_symbols_by_name,
            allow_unique_unqualified=False,
        )

    def symbols_for_node(self, node: ast.AST) -> tuple[str, ...]:
        collector = ClassReferenceSymbolCollector(self)
        collector.visit(node)
        return sorted_tuple(collector.symbols)

    def symbol_for_reference(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            return self._direct_constructor_symbol(node)
        if isinstance(node, ast.Name):
            constructor_symbol = self.constructor_assignment_symbols.get(node.id)
            if constructor_symbol is not None:
                return constructor_symbol
        return self.reference_resolution.symbol_for_node(node)

    def _direct_constructor_symbol(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        return self.reference_resolution.symbol_for_node(node.func)


class ClassReferenceSymbolCollector(ast.NodeVisitor):
    """Collect expression nodes that reference classes without counting members."""

    def __init__(self, resolver: ModuleClassReferenceResolver) -> None:
        self.resolver = resolver
        self.symbols: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        self._add_symbol(self.resolver._direct_constructor_symbol(node))
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_symbol(self.resolver.reference_resolution.symbol_for_node(node))

    def visit_Name(self, node: ast.Name) -> None:
        self._add_symbol(self.resolver.symbol_for_reference(node))

    def _add_symbol(self, symbol: str | None) -> None:
        if symbol is not None:
            self.symbols.add(symbol)


def _single_assignment_target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
    else:
        target = node.target
    if isinstance(target, ast.Name):
        return target.id
    return None


def build_class_family_index(modules: list[ParsedModule]) -> ClassFamilyIndex:
    return _build_class_family_index_cached(tuple(modules))


def overlay_class_family_index(
    base_index: ClassFamilyIndex,
    changed_modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    """Rebuild a class index by replacing changed-module class records."""

    changed_path_texts = _resolved_module_path_texts(changed_modules)
    unchanged_records = base_index.class_records_excluding_files(changed_path_texts)
    return ClassFamilyIndexBuilder(
        changed_modules,
        base_records=unchanged_records,
    ).build()


@lru_cache(maxsize=None)
def _build_class_family_index_cached(
    modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    return ClassFamilyIndexBuilder(modules).build()


@dataclass(frozen=True)
class ClassFamilyIndexBuilder:
    modules: tuple[ParsedModule, ...]
    base_records: tuple[IndexedClass, ...] = ()

    def build(self) -> ClassFamilyIndex:
        class_records = (*self.base_records, *self.module_class_records())
        known_symbols = frozenset(record.symbol for record in class_records)
        symbols_by_simple_name_multimap = self.symbols_by_simple_name_multimap(
            class_records
        )
        unique_symbols_by_name = {
            name: symbols[0]
            for name, symbols in symbols_by_simple_name_multimap.items()
            if len(symbols) == 1
        }
        classes_by_symbol = {
            record.symbol: self.resolved_record(
                record,
                known_symbols,
                unique_symbols_by_name,
            )
            for record in class_records
        }
        symbols_by_file_and_qualname = {
            (record.file_path, record.qualname): record.symbol
            for record in classes_by_symbol.values()
        }
        children_by_symbol = self.children_by_symbol(classes_by_symbol)
        ancestors_by_symbol = self.ancestors_by_symbol(classes_by_symbol)
        descendants_by_symbol = self.descendants_by_symbol(
            classes_by_symbol,
            children_by_symbol,
        )
        return ClassFamilyIndex(
            classes_by_symbol=classes_by_symbol,
            symbols_by_simple_name={
                name: sorted_tuple(symbols)
                for name, symbols in symbols_by_simple_name_multimap.items()
            },
            symbols_by_file_and_qualname=symbols_by_file_and_qualname,
            children_by_symbol=children_by_symbol,
            ancestors_by_symbol=ancestors_by_symbol,
            descendants_by_symbol=descendants_by_symbol,
        )

    def module_class_records(self) -> tuple[IndexedClass, ...]:
        records: list[IndexedClass] = []
        for parsed_module in self.modules:
            for qualname, node in _iter_class_defs(list(parsed_module.module.body)):
                records.append(
                    IndexedClass.from_parsed_class(parsed_module, qualname, node)
                )
        return tuple(records)

    @staticmethod
    def symbols_by_simple_name_multimap(
        class_records: tuple[IndexedClass, ...],
    ) -> dict[str, list[str]]:
        symbols_by_simple_name_multimap: dict[str, list[str]] = defaultdict(list)
        for record in class_records:
            symbols_by_simple_name_multimap[record.simple_name].append(record.symbol)
        return symbols_by_simple_name_multimap

    def resolved_record(
        self,
        record: IndexedClass,
        known_symbols: frozenset[str],
        unique_symbols_by_name: dict[str, str],
    ) -> IndexedClass:
        parsed_module = self.parsed_module_by_name.get(record.module_name)
        if parsed_module is None:
            return self.base_record_with_current_bases(record, known_symbols)
        base_resolution = ClassSymbolResolutionAuthority(
            parsed_module=parsed_module,
            import_aliases=_module_import_aliases(parsed_module),
            known_symbols=known_symbols,
            unique_symbols_by_name=unique_symbols_by_name,
            allow_unique_unqualified=True,
        )
        return record.with_resolved_base_symbols(
            tuple(
                resolved
                for base in record.node.bases
                if (resolved := base_resolution.symbol_for_node(base)) is not None
            )
        )

    @cached_property
    def parsed_module_by_name(self) -> dict[str, ParsedModule]:
        return {module.module_name: module for module in self.modules}

    @staticmethod
    def base_record_with_current_bases(
        record: IndexedClass,
        known_symbols: frozenset[str],
    ) -> IndexedClass:
        return record.with_resolved_base_symbols(
            tuple(
                base_symbol
                for base_symbol in record.resolved_base_symbols
                if base_symbol in known_symbols
            )
        )

    @staticmethod
    def children_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        children_by_symbol_lists: dict[str, list[str]] = defaultdict(list)
        for record in classes_by_symbol.values():
            for base_symbol in record.resolved_base_symbols:
                children_by_symbol_lists[base_symbol].append(record.symbol)
        return {
            symbol: sorted_tuple(children)
            for symbol, children in children_by_symbol_lists.items()
        }

    @staticmethod
    def ancestors_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
    ) -> dict[str, tuple[str, ...]]:
        ancestors_by_symbol: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            ancestors: list[str] = []
            queue = list(classes_by_symbol[symbol].resolved_base_symbols)
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                ancestors.append(current)
                indexed_class = classes_by_symbol.get(current)
                if indexed_class is not None:
                    queue.extend(indexed_class.resolved_base_symbols)
            if ancestors:
                ancestors_by_symbol[symbol] = tuple(ancestors)
        return ancestors_by_symbol

    @staticmethod
    def descendants_by_symbol(
        classes_by_symbol: dict[str, IndexedClass],
        children_by_symbol: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        descendants_by_symbol: dict[str, tuple[str, ...]] = {}
        for symbol in sorted(classes_by_symbol):
            descendants: list[str] = []
            queue = (
                list(children_by_symbol[symbol])
                if symbol in children_by_symbol
                else []
            )
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                descendants.append(current)
                if current in children_by_symbol:
                    queue.extend(children_by_symbol[current])
            if descendants:
                descendants_by_symbol[symbol] = tuple(descendants)
        return descendants_by_symbol


def _resolved_module_path_texts(modules: tuple[ParsedModule, ...]) -> frozenset[str]:
    return frozenset(_resolved_path_text(str(module.path)) for module in modules)


def _resolved_path_text(file_path: str) -> str:
    return str(Path(file_path).resolve())
