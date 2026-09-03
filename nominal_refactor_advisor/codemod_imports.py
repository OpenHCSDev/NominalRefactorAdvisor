"""Canonical import syntax for codemod source mutations."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import (
    dataclass,
    replace,
)
from typing import TYPE_CHECKING, cast

from .ast_tools import ImportBoundNameProjection
from .codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceInsertion,
    SourceSpanReplacement,
    SourceTargetEditor,
    _joined_rationales,
)
from .codemod_spacing import DestinationInsertionSpacing
from .collection_algebra import sorted_tuple

if TYPE_CHECKING:
    from .codemod import CodemodSelectorContext


@dataclass(frozen=True)
class ModuleImportInsertionPoint:
    """Insertion line after a module docstring and leading import block."""

    source: str
    file_path: str
    module_node: ast.Module | None = None

    @property
    def line_number(self) -> int:
        imports = self.leading_import_statements
        if imports:
            final_import = imports[-1]
            return (final_import.end_lineno or final_import.lineno) + 1
        body = self.module.body
        if not body:
            return 1
        if self._first_statement_index_after_docstring(body):
            docstring = body[0]
            return (docstring.end_lineno or docstring.lineno) + 1
        return 1

    @property
    def module(self) -> ast.Module:
        module = self.module_node
        if module is None:
            module = ast.parse(self.source, filename=self.file_path)
        return module

    @property
    def leading_import_statements(
        self,
    ) -> tuple[ast.Import | ast.ImportFrom, ...]:
        body = self.module.body
        if not body:
            return ()
        index = self._first_statement_index_after_docstring(body)
        imports: list[ast.Import | ast.ImportFrom] = []
        while index < len(body) and isinstance(
            body[index], (ast.Import, ast.ImportFrom)
        ):
            imports.append(body[index])
            index += 1
        return tuple(imports)

    @property
    def previous_import_statement(self) -> ast.Import | ast.ImportFrom | None:
        imports = self.leading_import_statements
        return imports[-1] if imports else None

    @staticmethod
    def _first_statement_index_after_docstring(body: list[ast.stmt]) -> int:
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return 1
        return 0


@dataclass(frozen=True)
class ImportNameRemoval:
    """Names removed from one nominal from-import module."""

    module_name: ImportFromModuleName
    names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ModuleImportMutation(NominalSourceEdit):
    """Typed additions and removals resolved once against a module import block."""

    file_path: str
    additions: tuple[RequestedImportStatement, ...] = ()
    removals: tuple[ImportNameRemoval, ...] = ()
    bound_name_removals: tuple[str, ...] = ()

    @classmethod
    def from_source(
        cls,
        *,
        file_path: str,
        import_source: str,
        rationale: str = "",
    ) -> "ModuleImportMutation":
        requested = RequestedImportStatement.from_source(import_source)
        if not requested:
            raise ValueError("Module import mutations require import statements")
        return cls(
            file_path=file_path,
            additions=requested,
            rationale=rationale,
        )

    @classmethod
    def remove_names(
        cls,
        *,
        file_path: str,
        module_name: str,
        names: Iterable[str],
        rationale: str = "",
    ) -> "ModuleImportMutation":
        return cls(
            file_path=file_path,
            removals=(
                ImportNameRemoval(
                    module_name=ImportFromModuleName(module_name),
                    names=tuple(dict.fromkeys(names)),
                ),
            ),
            rationale=rationale,
        )

    @classmethod
    def remove_bound_names(
        cls,
        *,
        file_path: str,
        names: Iterable[str],
        rationale: str = "",
    ) -> "ModuleImportMutation":
        """Remove exact import bindings after current-source resolution."""

        return cls(
            file_path=file_path,
            bound_name_removals=tuple(dict.fromkeys(names)),
            rationale=rationale,
        )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        mutations_by_path: dict[str, list[ModuleImportMutation]] = defaultdict(list)
        for peer in peers:
            mutation = cast(ModuleImportMutation, peer)
            mutations_by_path[mutation.file_path].append(mutation)
        return tuple(
            self._coalesced_file_mutation(tuple(mutations))
            for mutations in mutations_by_path.values()
        )

    @classmethod
    def _coalesced_file_mutation(
        cls,
        mutations: tuple["ModuleImportMutation", ...],
    ) -> "ModuleImportMutation":
        first = mutations[0]
        additions = cls._coalesced_additions(
            addition for mutation in mutations for addition in mutation.additions
        )
        removals = cls._coalesced_removals(
            removal for mutation in mutations for removal in mutation.removals
        )
        bound_name_removals = tuple(
            sorted(
                {
                    name
                    for mutation in mutations
                    for name in mutation.bound_name_removals
                }
            )
        )
        removed_names_by_module = {
            removal.module_name: frozenset(removal.names) for removal in removals
        }
        conflicts = tuple(
            (addition.module_name.source, alias.name)
            for addition in additions
            if addition.module_name is not None
            for alias in addition.aliases
            if alias.name
            in removed_names_by_module.get(
                addition.module_name,
                frozenset(),
            )
        )
        bound_name_conflicts = tuple(
            sorted(
                {
                    name
                    for addition in additions
                    for name in addition.bound_names
                    if name in bound_name_removals
                }
            )
        )
        if conflicts or bound_name_conflicts:
            raise ValueError(
                "Import mutations both add and remove names: "
                f"{(*conflicts, *bound_name_conflicts)!r}"
            )
        return replace(
            first,
            additions=additions,
            removals=removals,
            bound_name_removals=bound_name_removals,
            rationale=_joined_rationales(mutation.rationale for mutation in mutations),
            contributors=NominalSourceEdit.merged_contributors(mutations),
            origins=NominalSourceEdit.merged_origins(mutations),
        )

    @staticmethod
    def _coalesced_additions(
        additions: Iterable[RequestedImportStatement],
    ) -> tuple[RequestedImportStatement, ...]:
        aliases_by_family: dict[
            tuple[type[ast.Import | ast.ImportFrom], int, str | None],
            list[ImportAliasRequirement],
        ] = {}
        statement_by_family: dict[
            tuple[type[ast.Import | ast.ImportFrom], int, str | None],
            RequestedImportStatement,
        ] = {}
        for addition in additions:
            family = addition.family_identity
            statement_by_family.setdefault(family, addition)
            aliases = aliases_by_family.setdefault(family, [])
            for alias in addition.aliases:
                if alias not in aliases:
                    aliases.append(alias)
        return tuple(
            statement_by_family[family].with_aliases(
                sorted_tuple(
                    aliases_by_family[family],
                    key=lambda alias: alias.canonical_key,
                )
            )
            for family in sorted(
                statement_by_family,
                key=lambda item: statement_by_family[item].canonical_family_key,
            )
        )

    @staticmethod
    def _coalesced_removals(
        removals: Iterable[ImportNameRemoval],
    ) -> tuple[ImportNameRemoval, ...]:
        names_by_module: dict[ImportFromModuleName, list[str]] = {}
        for removal in removals:
            names = names_by_module.setdefault(removal.module_name, [])
            for name in removal.names:
                if name not in names:
                    names.append(name)
        return tuple(
            ImportNameRemoval(
                module_name=module_name,
                names=tuple(sorted(names)),
            )
            for module_name, names in sorted(
                names_by_module.items(),
                key=lambda item: item[0].source,
            )
        )

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[PhysicalSourceEdit, ...]:
        source = context.sources_by_file_path[self.file_path]
        module = context.module_nodes_by_file_path.get(self.file_path)
        if module is None:
            module = ast.parse(source, filename=self.file_path)
        additions = list(self._coalesced_additions(self.additions))
        removals_by_module = {
            removal.module_name: frozenset(removal.names)
            for removal in self._coalesced_removals(self.removals)
        }
        requested_bound_name_removals = frozenset(self.bound_name_removals)
        import_statements = tuple(
            statement
            for statement in module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        )
        import_from_statements = tuple(
            statement
            for statement in import_statements
            if isinstance(statement, ast.ImportFrom)
        )
        aliases_by_statement = {
            id(statement): [
                ImportAliasRequirement.from_alias(alias) for alias in statement.names
            ]
            for statement in import_statements
        }

        for statement in import_from_statements:
            module_name = ImportFromModuleName.from_node(statement)
            removed_names = removals_by_module.get(module_name, frozenset())
            if removed_names and any(alias.name == "*" for alias in statement.names):
                raise ValueError(
                    f"Cannot remove named imports from star import {module_name.source!r}"
                )
            aliases_by_statement[id(statement)] = [
                alias
                for alias in aliases_by_statement[id(statement)]
                if alias.name not in removed_names
            ]

        matched_bound_name_removals: set[str] = set()
        for statement in import_statements:
            requested_statement = RequestedImportStatement(statement)
            retained_aliases: list[ImportAliasRequirement] = []
            for alias in aliases_by_statement[id(statement)]:
                bound_name = requested_statement.bound_name(alias)
                if (
                    bound_name is not None
                    and bound_name in requested_bound_name_removals
                ):
                    matched_bound_name_removals.add(bound_name)
                else:
                    retained_aliases.append(alias)
            aliases_by_statement[id(statement)] = retained_aliases
        unmatched_bound_name_removals = tuple(
            sorted(requested_bound_name_removals - matched_bound_name_removals)
        )
        if unmatched_bound_name_removals:
            raise ValueError(
                "Import bindings no longer resolve for removal: "
                f"{unmatched_bound_name_removals!r}"
            )

        pending_additions: list[RequestedImportStatement] = []
        for addition in additions:
            matching_from_statements = tuple(
                statement
                for statement in import_from_statements
                if addition.module_name == ImportFromModuleName.from_node(statement)
            )
            if addition.module_name is None:
                existing_aliases = tuple(
                    alias
                    for statement in import_statements
                    if isinstance(statement, ast.Import)
                    for alias in aliases_by_statement[id(statement)]
                )
                missing_aliases = tuple(
                    alias for alias in addition.aliases if alias not in existing_aliases
                )
                if not missing_aliases:
                    continue
                pending_additions.append(addition.with_aliases(missing_aliases))
                continue
            if any(
                alias.name == "*"
                for statement in matching_from_statements
                for alias in aliases_by_statement[id(statement)]
            ):
                continue
            if not matching_from_statements:
                pending_additions.append(addition)
                continue
            target_statement = matching_from_statements[0]
            aliases = aliases_by_statement[id(target_statement)]
            existing_aliases = tuple(
                alias
                for statement in matching_from_statements
                for alias in aliases_by_statement[id(statement)]
            )
            for alias in addition.aliases:
                if alias in existing_aliases:
                    continue
                if alias not in aliases:
                    aliases.append(alias)

        replacements: list[PhysicalSourceEdit] = []
        for statement in import_statements:
            original_aliases = tuple(
                ImportAliasRequirement.from_alias(alias) for alias in statement.names
            )
            aliases = tuple(aliases_by_statement[id(statement)])
            if aliases == original_aliases:
                continue
            replacement = RequestedImportStatement(statement).with_aliases(aliases)
            replacement_source = replacement.source if aliases else ""
            replacements.append(
                SourceSpanReplacement(
                    file_path=self.file_path,
                    start_line=statement.lineno,
                    end_line=statement.end_lineno or statement.lineno,
                    replacement_lines=SourceTargetEditor.source_lines(
                        replacement_source
                    ),
                    rationale=self.rationale
                    or f"Update import bindings in {self.file_path!r}.",
                    contributors=self.contributors,
                    origins=self.origins,
                )
            )
        if pending_additions:
            insertion_point = ModuleImportInsertionPoint(
                source,
                self.file_path,
                module_node=module,
            )
            insertion_line = insertion_point.line_number
            spacing = DestinationInsertionSpacing.from_source(
                source,
                insertion_line,
                inserted_source_is_import_block=True,
            )
            replacements.append(
                SourceInsertion(
                    file_path=self.file_path,
                    insertion_line=insertion_line,
                    inserted_lines=SourceTargetEditor.source_lines(
                        RequestedImportBlock(tuple(pending_additions)).source_after(
                            insertion_point.previous_import_statement
                        )
                        + spacing.trailing_separator
                    ),
                    rationale=self.rationale
                    or f"Ensure imports exist in {self.file_path!r}.",
                    contributors=self.contributors,
                    origins=self.origins,
                )
            )
        return tuple(replacements)


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
    def canonical_family_key(self) -> tuple[bool, bool, str, int, str]:
        """Return deterministic source order without encoding semantic priority."""

        if isinstance(self.statement, ast.Import):
            return True, False, type(self.statement).__name__, 0, ""
        return (
            not self.is_future_import,
            self.is_relative_import,
            type(self.statement).__name__,
            self.statement.level,
            self.statement.module or "",
        )

    @property
    def is_future_import(self) -> bool:
        """Return whether this statement belongs to Python's future-import group."""

        return (
            isinstance(self.statement, ast.ImportFrom)
            and self.statement.level == 0
            and self.statement.module == "__future__"
        )

    @property
    def is_relative_import(self) -> bool:
        """Return whether this statement belongs to the relative-import group."""

        return isinstance(self.statement, ast.ImportFrom) and self.statement.level > 0

    @property
    def source_group_identity(self) -> tuple[bool, bool]:
        """Return the syntax-derived import-block group identity."""

        return self.is_future_import, self.is_relative_import

    @property
    def bound_names(self) -> tuple[str, ...]:
        """Return names introduced into the importing module."""

        return tuple(
            bound_name
            for alias in self.aliases
            if (bound_name := self.bound_name(alias)) is not None
        )

    def bound_name(self, alias: ImportAliasRequirement) -> str | None:
        bound_names = ImportBoundNameProjection(
            self.with_aliases((alias,)).statement
        ).names()
        return bound_names[0] if bound_names else None

    @property
    def source(self) -> str:
        if isinstance(self.statement, ast.Import):
            return "".join(
                f"import {ImportFromSource.alias_source(alias)}\n"
                for alias in self.statement.names
            )
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
class RequestedImportBlock:
    """Canonical source for a group of requested import statements."""

    statements: tuple[RequestedImportStatement, ...]

    def source_after(
        self,
        previous_statement: ast.Import | ast.ImportFrom | None,
    ) -> str:
        previous = (
            RequestedImportStatement(previous_statement)
            if previous_statement is not None
            else None
        )
        source_parts: list[str] = []
        for statement in self.statements:
            if (
                previous is not None
                and previous.source_group_identity != statement.source_group_identity
            ):
                source_parts.append("\n")
            source_parts.append(statement.source)
            previous = statement
        return "".join(source_parts)


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
