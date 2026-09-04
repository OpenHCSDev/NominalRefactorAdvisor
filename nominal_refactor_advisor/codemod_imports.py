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
from .codemod_import_scopes import (
    ImportStatement,
    ModuleImportScope,
    TypeCheckingGuardProjection,
)
from .codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTargetEditor,
    SourceTextSpan,
    _joined_rationales,
)
from .codemod_spacing import DestinationInsertionSpacing
from .collection_algebra import sorted_tuple

if TYPE_CHECKING:
    from .codemod_selection_context import CodemodSelectorContext


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
class TypeCheckingGuardImportInsertionPoint:
    """Insertion geometry derived from one existing type-checking guard."""

    source: str
    guard: ast.If

    @property
    def leading_import_statements(self) -> tuple[ImportStatement, ...]:
        imports: list[ImportStatement] = []
        for statement in self.guard.body:
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                break
            imports.append(statement)
        return tuple(imports)

    @property
    def previous_import_statement(self) -> ImportStatement | None:
        imports = self.leading_import_statements
        return imports[-1] if imports else None

    @property
    def line_number(self) -> int:
        previous = self.previous_import_statement
        if previous is not None:
            return (previous.end_lineno or previous.lineno) + 1
        return self.guard.body[0].lineno

    @property
    def indentation(self) -> str:
        return _line_indentation(self.source, self.guard.body[0].lineno)

    def indented_source(
        self,
        additions: tuple[RequestedImportStatement, ...],
    ) -> str:
        source = _indent_source(
            RequestedImportBlock(additions).source_after(
                self.previous_import_statement
            ),
            self.indentation,
        )
        if len(self.leading_import_statements) < len(self.guard.body):
            return f"{source}\n"
        return source


def _line_indentation(source: str, line_number: int) -> str:
    line = source.splitlines(keepends=True)[line_number - 1]
    return line[: len(line) - len(line.lstrip())]


def _indent_source(source: str, indentation: str) -> str:
    return "".join(
        f"{indentation}{line}" if line.strip() else line
        for line in source.splitlines(keepends=True)
    )


def _indented_source_for_statement(
    source: str,
    statement: ast.stmt,
    replacement_source: str,
) -> str:
    return _indent_source(
        replacement_source,
        _line_indentation(source, statement.lineno),
    )


def _node_contains_comment(source: str, node: ast.AST) -> bool:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return False
    fragment = SourceLineSpan(
        start_line=node.lineno,
        end_line=node.end_lineno or node.lineno,
    ).source_from(source)
    return SourceTextSpan(
        start_offset=0,
        end_offset=len(fragment),
    ).contains_comment(fragment)


@dataclass(frozen=True)
class ImportNameRemoval:
    """Names removed from one nominal from-import module."""

    module_name: ImportFromModuleName
    names: tuple[str, ...]
    scope: ModuleImportScope = ModuleImportScope.RUNTIME


@dataclass(frozen=True)
class ImportBoundNameRemoval:
    """One exact import binding removed from its declared execution scope."""

    name: str
    scope: ModuleImportScope = ModuleImportScope.RUNTIME


@dataclass(frozen=True, kw_only=True)
class ModuleImportMutation(NominalSourceEdit):
    """Typed additions and removals resolved once across module import scopes."""

    file_path: str
    additions: tuple[RequestedImportStatement, ...] = ()
    removals: tuple[ImportNameRemoval, ...] = ()
    bound_name_removals: tuple[ImportBoundNameRemoval, ...] = ()

    @classmethod
    def from_source(
        cls,
        *,
        file_path: str,
        import_source: str,
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
        rationale: str = "",
    ) -> "ModuleImportMutation":
        requested = RequestedImportStatement.from_source(import_source, scope=scope)
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
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
        rationale: str = "",
    ) -> "ModuleImportMutation":
        return cls(
            file_path=file_path,
            removals=(
                ImportNameRemoval(
                    module_name=ImportFromModuleName(module_name),
                    names=tuple(dict.fromkeys(names)),
                    scope=scope,
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
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
        rationale: str = "",
    ) -> "ModuleImportMutation":
        """Remove exact import bindings after current-source resolution."""

        return cls(
            file_path=file_path,
            bound_name_removals=tuple(
                ImportBoundNameRemoval(name=name, scope=scope)
                for name in dict.fromkeys(names)
            ),
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
                    removal
                    for mutation in mutations
                    for removal in mutation.bound_name_removals
                },
                key=lambda removal: (removal.scope.canonical_rank, removal.name),
            )
        )
        removed_names_by_module = {
            (removal.scope, removal.module_name): frozenset(removal.names)
            for removal in removals
        }
        conflicts = tuple(
            (addition.scope.value, addition.module_name.source, alias.name)
            for addition in additions
            if addition.module_name is not None
            for alias in addition.aliases
            if alias.name
            in removed_names_by_module.get(
                (addition.scope, addition.module_name),
                frozenset(),
            )
        )
        bound_name_conflicts = tuple(
            sorted(
                {
                    (addition.scope.value, removal.name)
                    for addition in additions
                    for removal in bound_name_removals
                    if addition.scope is removal.scope
                    and removal.name in addition.bound_names
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
            tuple[ModuleImportScope, type[ImportStatement], int, str | None],
            list[ImportAliasRequirement],
        ] = {}
        statement_by_family: dict[
            tuple[ModuleImportScope, type[ImportStatement], int, str | None],
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
        names_by_module: dict[
            tuple[ModuleImportScope, ImportFromModuleName],
            list[str],
        ] = {}
        for removal in removals:
            names = names_by_module.setdefault(
                (removal.scope, removal.module_name),
                [],
            )
            for name in removal.names:
                if name not in names:
                    names.append(name)
        return tuple(
            ImportNameRemoval(
                module_name=module_name,
                names=tuple(sorted(names)),
                scope=scope,
            )
            for (scope, module_name), names in sorted(
                names_by_module.items(),
                key=lambda item: (
                    item[0][0].canonical_rank,
                    item[0][1].source,
                ),
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
            (removal.scope, removal.module_name): frozenset(removal.names)
            for removal in self._coalesced_removals(self.removals)
        }
        requested_bound_name_removals = frozenset(self.bound_name_removals)
        import_statements = tuple(
            statement
            for scope in ModuleImportScope
            for statement in scope.import_statements(module)
        )
        scope_by_statement_id = {
            id(statement): scope
            for scope in ModuleImportScope
            for statement in scope.import_statements(module)
        }
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
            scope = scope_by_statement_id[id(statement)]
            module_name = ImportFromModuleName.from_node(statement)
            removed_names = removals_by_module.get(
                (scope, module_name),
                frozenset(),
            )
            if removed_names and any(alias.name == "*" for alias in statement.names):
                raise ValueError(
                    f"Cannot remove named imports from star import {module_name.source!r}"
                )
            aliases_by_statement[id(statement)] = [
                alias
                for alias in aliases_by_statement[id(statement)]
                if alias.name not in removed_names
            ]

        matched_bound_name_removals: set[ImportBoundNameRemoval] = set()
        for statement in import_statements:
            scope = scope_by_statement_id[id(statement)]
            requested_statement = RequestedImportStatement(statement, scope=scope)
            retained_aliases: list[ImportAliasRequirement] = []
            for alias in aliases_by_statement[id(statement)]:
                bound_name = requested_statement.bound_name(alias)
                removal = (
                    ImportBoundNameRemoval(name=bound_name, scope=scope)
                    if bound_name is not None
                    else None
                )
                if removal is not None and removal in requested_bound_name_removals:
                    matched_bound_name_removals.add(removal)
                else:
                    retained_aliases.append(alias)
            aliases_by_statement[id(statement)] = retained_aliases
        unmatched_bound_name_removals = tuple(
            sorted(requested_bound_name_removals - matched_bound_name_removals)
        )
        if unmatched_bound_name_removals:
            raise ValueError(
                "Import bindings no longer resolve for removal: "
                f"{tuple((item.scope.value, item.name) for item in unmatched_bound_name_removals)!r}"
            )

        pending_additions: list[RequestedImportStatement] = []
        for addition in additions:
            matching_from_statements = tuple(
                statement
                for statement in import_from_statements
                if scope_by_statement_id[id(statement)] is addition.scope
                if addition.module_name == ImportFromModuleName.from_node(statement)
            )
            if addition.module_name is None:
                existing_aliases = tuple(
                    alias
                    for statement in import_statements
                    if isinstance(statement, ast.Import)
                    and scope_by_statement_id[id(statement)] is addition.scope
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
        guard_projection = TypeCheckingGuardProjection.from_module(module)
        guard_by_statement_id = {
            id(statement): guard
            for guard in guard_projection.guards
            for statement in guard.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
        }
        has_guarded_pending_additions = any(
            addition.scope.is_guarded for addition in pending_additions
        )
        guarded_addition_target = (
            guard_projection.guards[0]
            if has_guarded_pending_additions and guard_projection.guards
            else None
        )
        deletable_guards = frozenset(
            id(guard)
            for guard in guard_projection.guards
            if guard is not guarded_addition_target
            and not guard.orelse
            and guard.body
            and all(
                isinstance(statement, (ast.Import, ast.ImportFrom))
                and not aliases_by_statement[id(statement)]
                for statement in guard.body
            )
            and not _node_contains_comment(source, guard)
        )
        preserved_empty_guard_statement_ids = frozenset(
            id(guard.body[-1])
            for guard in guard_projection.guards
            if guard is not guarded_addition_target
            and guard.body
            and id(guard) not in deletable_guards
            and all(
                isinstance(statement, (ast.Import, ast.ImportFrom))
                and not aliases_by_statement[id(statement)]
                for statement in guard.body
            )
        )
        replacements.extend(
            SourceSpanDeletion(
                file_path=self.file_path,
                start_line=guard.lineno,
                end_line=guard.end_lineno or guard.lineno,
                rationale=self.rationale
                or f"Remove empty type-checking import guard in {self.file_path!r}.",
                contributors=self.contributors,
                origins=self.origins,
            )
            for guard in guard_projection.guards
            if id(guard) in deletable_guards
        )
        for statement in import_statements:
            guard = guard_by_statement_id.get(id(statement))
            if guard is not None and id(guard) in deletable_guards:
                continue
            original_aliases = tuple(
                ImportAliasRequirement.from_alias(alias) for alias in statement.names
            )
            aliases = tuple(aliases_by_statement[id(statement)])
            if aliases == original_aliases:
                continue
            scope = scope_by_statement_id[id(statement)]
            if not aliases and id(statement) not in preserved_empty_guard_statement_ids:
                replacements.append(
                    SourceSpanDeletion(
                        file_path=self.file_path,
                        start_line=statement.lineno,
                        end_line=statement.end_lineno or statement.lineno,
                        rationale=self.rationale
                        or f"Remove import bindings in {self.file_path!r}.",
                        contributors=self.contributors,
                        origins=self.origins,
                    )
                )
                continue
            replacement_source = (
                "pass\n"
                if id(statement) in preserved_empty_guard_statement_ids
                else RequestedImportStatement(statement, scope=scope)
                .with_aliases(aliases)
                .source
            )
            if scope.is_guarded and replacement_source:
                replacement_source = _indented_source_for_statement(
                    source,
                    statement,
                    replacement_source,
                )
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
            replacements.extend(
                self._pending_addition_edits(
                    source=source,
                    module=module,
                    pending_additions=tuple(pending_additions),
                    guard_projection=guard_projection,
                )
            )
        return tuple(replacements)

    def _pending_addition_edits(
        self,
        *,
        source: str,
        module: ast.Module,
        pending_additions: tuple[RequestedImportStatement, ...],
        guard_projection: TypeCheckingGuardProjection,
    ) -> tuple[SourceInsertion, ...]:
        runtime_additions = tuple(
            addition for addition in pending_additions if not addition.scope.is_guarded
        )
        guarded_additions = tuple(
            addition for addition in pending_additions if addition.scope.is_guarded
        )
        if guarded_additions and not guard_projection.guards:
            return (
                self._new_guard_insertion(
                    source=source,
                    module=module,
                    runtime_additions=runtime_additions,
                    guarded_additions=guarded_additions,
                    guard_projection=guard_projection,
                ),
            )
        insertions: list[SourceInsertion] = []
        if runtime_additions:
            insertions.append(
                self._runtime_import_insertion(source, module, runtime_additions)
            )
        if guarded_additions:
            insertions.append(
                self._existing_guard_insertion(
                    source,
                    guard_projection.guards[0],
                    guarded_additions,
                )
            )
        return tuple(insertions)

    def _runtime_import_insertion(
        self,
        source: str,
        module: ast.Module,
        additions: tuple[RequestedImportStatement, ...],
    ) -> SourceInsertion:
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
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                RequestedImportBlock(additions).source_after(
                    insertion_point.previous_import_statement
                )
                + spacing.trailing_separator
            ),
            rationale=self.rationale or f"Ensure imports exist in {self.file_path!r}.",
            contributors=self.contributors,
            origins=self.origins,
        )

    def _existing_guard_insertion(
        self,
        source: str,
        guard: ast.If,
        additions: tuple[RequestedImportStatement, ...],
    ) -> SourceInsertion:
        insertion_point = TypeCheckingGuardImportInsertionPoint(source, guard)
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=insertion_point.line_number,
            inserted_lines=SourceTargetEditor.source_lines(
                insertion_point.indented_source(additions)
            ),
            rationale=self.rationale
            or f"Ensure type-checking imports exist in {self.file_path!r}.",
            contributors=self.contributors,
            origins=self.origins,
        )

    def _new_guard_insertion(
        self,
        *,
        source: str,
        module: ast.Module,
        runtime_additions: tuple[RequestedImportStatement, ...],
        guarded_additions: tuple[RequestedImportStatement, ...],
        guard_projection: TypeCheckingGuardProjection,
    ) -> SourceInsertion:
        reference_source = guard_projection.preferred_reference_source
        if reference_source is None:
            reference_source = "TYPE_CHECKING"
            runtime_additions = self._coalesced_additions(
                (
                    *runtime_additions,
                    *RequestedImportStatement.from_source(
                        "from typing import TYPE_CHECKING\n"
                    ),
                )
            )
        insertion_point = ModuleImportInsertionPoint(
            source,
            self.file_path,
            module_node=module,
        )
        runtime_source = RequestedImportBlock(runtime_additions).source_after(
            insertion_point.previous_import_statement
        )
        guarded_source = RequestedImportBlock(guarded_additions).source_after(None)
        separator_before_guard = (
            "\n"
            if runtime_source or insertion_point.previous_import_statement is not None
            else ""
        )
        inserted_source = (
            f"{runtime_source}{separator_before_guard}if {reference_source}:\n"
            f"{_indent_source(guarded_source, '    ')}"
        )
        insertion_line = insertion_point.line_number
        spacing = DestinationInsertionSpacing.from_source(
            source,
            insertion_line,
            inserted_source_is_import_block=False,
        )
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                inserted_source.rstrip("\n") + spacing.trailing_separator
            ),
            rationale=self.rationale
            or f"Ensure type-checking imports exist in {self.file_path!r}.",
            contributors=self.contributors,
            origins=self.origins,
        )


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
    scope: ModuleImportScope = ModuleImportScope.RUNTIME

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
    ) -> tuple["RequestedImportStatement", ...]:
        module = ast.parse(source, filename="<requested-import>")
        statements = tuple(
            cls(statement, scope=scope)
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
    ) -> tuple[ModuleImportScope, type[ImportStatement], int, str | None]:
        if isinstance(self.statement, ast.Import):
            return self.scope, ast.Import, 0, None
        return self.scope, ast.ImportFrom, self.statement.level, self.statement.module

    @property
    def canonical_family_key(self) -> tuple[int, bool, bool, str, int, str]:
        """Return deterministic source order without encoding semantic priority."""

        if isinstance(self.statement, ast.Import):
            return (
                self.scope.canonical_rank,
                True,
                False,
                type(self.statement).__name__,
                0,
                "",
            )
        return (
            self.scope.canonical_rank,
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
            return RequestedImportStatement(
                ast.Import(names=alias_nodes),
                scope=self.scope,
            )
        return RequestedImportStatement(
            ast.ImportFrom(
                module=self.statement.module,
                names=alias_nodes,
                level=self.statement.level,
            ),
            scope=self.scope,
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
