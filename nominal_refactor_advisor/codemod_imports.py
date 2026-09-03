"""Canonical import syntax for codemod source mutations."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ImportAliasRequirement:
    """One requested import alias, including alias spelling when present."""

    name: str
    asname: str | None

    @classmethod
    def from_alias(cls, alias: ast.alias) -> "ImportAliasRequirement":
        return cls(name=alias.name, asname=alias.asname)

    @property
    def canonical_key(self) -> tuple[str, str]:
        """Return the source-spelling key for commutative import merging."""

        return self.name, self.asname or ""


@dataclass(frozen=True)
class RequestedImportStatement:
    """AST-normalized import requirement for idempotent import insertion."""

    statement: ast.Import | ast.ImportFrom

    @classmethod
    def from_source(cls, source: str) -> tuple["RequestedImportStatement", ...]:
        module = ast.parse(source, filename="<requested-import>")
        statements = tuple(
            cls(statement)
            for statement in module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        )
        if len(statements) != len(module.body):
            return ()
        return statements

    @property
    def aliases(self) -> tuple[ImportAliasRequirement, ...]:
        return tuple(
            ImportAliasRequirement.from_alias(alias) for alias in self.statement.names
        )

    @property
    def module_name(self) -> "ImportFromModuleName | None":
        if not isinstance(self.statement, ast.ImportFrom):
            return None
        return ImportFromModuleName.from_node(self.statement)

    @property
    def family_identity(
        self,
    ) -> tuple[type[ast.Import | ast.ImportFrom], int, str | None]:
        if isinstance(self.statement, ast.Import):
            return ast.Import, 0, None
        return ast.ImportFrom, self.statement.level, self.statement.module

    @property
    def canonical_family_key(self) -> tuple[str, int, str]:
        """Return deterministic source order without encoding semantic priority."""

        if isinstance(self.statement, ast.Import):
            return type(self.statement).__name__, 0, ""
        return (
            type(self.statement).__name__,
            self.statement.level,
            self.statement.module or "",
        )

    @property
    def source(self) -> str:
        if isinstance(self.statement, ast.Import):
            aliases = ", ".join(
                ImportFromSource.alias_source(alias) for alias in self.statement.names
            )
            return f"import {aliases}\n"
        return ImportFromSource(
            module_name=ImportFromModuleName.from_node(self.statement).source,
            aliases=tuple(self.statement.names),
        ).source

    def with_aliases(
        self,
        aliases: Iterable[ImportAliasRequirement],
    ) -> "RequestedImportStatement":
        alias_nodes = [
            ast.alias(name=alias.name, asname=alias.asname) for alias in aliases
        ]
        if isinstance(self.statement, ast.Import):
            return RequestedImportStatement(ast.Import(names=alias_nodes))
        return RequestedImportStatement(
            ast.ImportFrom(
                module=self.statement.module,
                names=alias_nodes,
                level=self.statement.level,
            )
        )


@dataclass(frozen=True)
class ImportFromModuleName:
    """Canonical source spelling for an ImportFrom module."""

    source: str

    @classmethod
    def from_node(cls, node: ast.ImportFrom) -> "ImportFromModuleName":
        relative_prefix = "." * node.level
        if node.module is None:
            return cls(relative_prefix)
        return cls(f"{relative_prefix}{node.module}")


@dataclass(frozen=True)
class ImportFromSource:
    """Rendered from-import source for remaining aliases."""

    module_name: str
    aliases: tuple[ast.alias, ...]

    @property
    def source(self) -> str:
        if not self.aliases:
            return ""
        if len(self.aliases) == 1:
            return f"from {self.module_name} import {self.alias_sources[0]}\n"
        alias_lines = "".join(
            f"    {alias_source},\n" for alias_source in self.alias_sources
        )
        return f"from {self.module_name} import (\n{alias_lines})\n"

    @property
    def alias_sources(self) -> tuple[str, ...]:
        return tuple(self.alias_source(alias) for alias in self.aliases)

    @staticmethod
    def alias_source(alias: ast.alias) -> str:
        if alias.asname is None:
            return alias.name
        return f"{alias.name} as {alias.asname}"
