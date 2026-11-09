"""Canonical import syntax for codemod source mutations."""

from __future__ import annotations

import ast
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import (
    dataclass,
    replace,
)
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from nominal_refactor_advisor.codemod_import_scopes import (
    ImportStatement,
    ModuleImportScope,
    TypeCheckingGuardProjection,
)
from nominal_refactor_advisor.codemod_source_edits import (
    KeyedSourceEditCoalescence,
    PhysicalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanDeletion,
    SourceSpanReplacement,
    SourceTextGeometry,
    SourceTextSpan,
)
from nominal_refactor_advisor.codemod_spacing import DestinationInsertionSpacing
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.lexical_bindings import (
    ImportAlias as ImportAlias,
    ImportBoundNameProjection,
    ImportDeclarationABC,
    ImportFromModuleName as ImportFromModuleName,
)

if TYPE_CHECKING:
    from nominal_refactor_advisor.codemod_selection_context import (
        CodemodSelectorContext,
    )


class ImportBlockInsertionPointABC(ABC):
    """Shared insertion geometry for one contiguous Python import block."""

    file_path: str

    @property
    @abstractmethod
    def scope(self) -> ModuleImportScope:
        """Return the execution scope governing this import block."""

        raise NotImplementedError

    @property
    @abstractmethod
    def initial_line_number(self) -> int:
        """Return the source line preceding the first possible insertion."""

        raise NotImplementedError

    @property
    @abstractmethod
    def leading_import_statements(self) -> tuple[ImportStatement, ...]:
        """Return the imports governed by this insertion point, in source order."""

        raise NotImplementedError

    @property
    def line_number(self) -> int:
        """Return the source anchor following the final governed import."""

        imports = self.leading_import_statements
        if imports:
            final_import = imports[-1]
            return (final_import.end_lineno or final_import.lineno) + 1
        return self.initial_line_number

    def insertion_index_for(self, addition: RequestedImportStatement) -> int:
        """Locate the addition among existing canonical import families."""

        return next(
            (
                index
                for index, statement in enumerate(self.leading_import_statements)
                if RequestedImportStatement(
                    statement,
                    scope=self.scope,
                ).in_file(self.file_path).canonical_family_key
                > addition.in_file(self.file_path).canonical_family_key
            ),
            len(self.leading_import_statements),
        )

    @property
    def previous_import_statement(self) -> ImportStatement | None:
        """Return the final governed import, when one exists."""

        imports = self.leading_import_statements
        return imports[-1] if imports else None

    def source_for_insertion(
        self,
        *,
        source: str,
        slot: ImportInsertionSlot,
        additions: tuple[RequestedImportStatement, ...],
    ) -> str:
        """Render additions with any required following group boundary."""

        existing_imports = self.leading_import_statements
        inserted_source = RequestedImportBlock(additions, self.file_path).source_after(
            slot.previous_statement(self)
        )
        if (
            slot.index < len(existing_imports)
            and additions[-1].in_file(self.file_path).source_group
            is not RequestedImportStatement(existing_imports[slot.index])
            .in_file(self.file_path)
            .source_group
        ):
            spacing = DestinationInsertionSpacing.from_source(
                source,
                slot.line_number(self),
                inserted_source_is_import_block=True,
            )
            inserted_source += "\n" * max(
                0,
                1 - spacing.following_blank_line_count,
            )
        return inserted_source

    def grouped_additions(
        self,
        additions: tuple[RequestedImportStatement, ...],
    ) -> tuple[
        tuple[ImportInsertionSlot, tuple[RequestedImportStatement, ...]], ...
    ]:
        """Group additions by the slot each lands in, in source order."""

        additions_by_slot: dict[
            ImportInsertionSlot,
            list[RequestedImportStatement],
        ] = defaultdict(list)
        for addition in additions:
            slot = ImportInsertionSlot.of(
                self, self.insertion_index_for(addition), addition
            )
            additions_by_slot[slot].append(addition)
        return tuple(
            (slot, tuple(additions_by_slot[slot]))
            for slot in sorted(
                additions_by_slot, key=lambda slot: slot.line_number(self)
            )
        )


@dataclass(frozen=True)
class ImportInsertionSlot(ABC):
    """Where one group of additions lands in an import block: before the existing
    import at ``index``, the first whose family sorts after them, either ending the
    preceding import's family or starting the next import's."""

    index: int

    @staticmethod
    def of(
        point: ImportBlockInsertionPointABC,
        index: int,
        addition: RequestedImportStatement,
    ) -> ImportInsertionSlot:
        imports = point.leading_import_statements
        if (
            index < len(imports)
            and RequestedImportStatement(imports[index], scope=point.scope)
            .in_file(point.file_path)
            .source_group
            is addition.in_file(point.file_path).source_group
        ):
            return StartOfNextFamily(index)
        return AfterPreviousImport(index)

    @abstractmethod
    def line_number(self, point: ImportBlockInsertionPointABC) -> int:
        """Return the source line the additions are inserted before."""

    @abstractmethod
    def previous_statement(
        self, point: ImportBlockInsertionPointABC
    ) -> ImportStatement | None:
        """Return the import the additions follow, for their leading group boundary."""


class AfterPreviousImport(ImportInsertionSlot):
    """After the preceding import: the additions end its family or open their own."""

    def line_number(self, point: ImportBlockInsertionPointABC) -> int:
        if self.index == 0:
            return point.initial_line_number
        previous = point.leading_import_statements[self.index - 1]
        return (previous.end_lineno or previous.lineno) + 1

    def previous_statement(
        self, point: ImportBlockInsertionPointABC
    ) -> ImportStatement | None:
        return point.leading_import_statements[self.index - 1] if self.index else None


class StartOfNextFamily(ImportInsertionSlot):
    """Directly before the next import, whose family the additions belong to and now
    begin, so the boundary before that family stays where it is."""

    def line_number(self, point: ImportBlockInsertionPointABC) -> int:
        return point.leading_import_statements[self.index].lineno

    def previous_statement(
        self, point: ImportBlockInsertionPointABC
    ) -> ImportStatement | None:
        return None


@dataclass(frozen=True)
class ModuleImportInsertionPoint(ImportBlockInsertionPointABC):
    """Insertion line after a module docstring and leading import block."""

    source: str
    file_path: str
    module_node: ast.Module | None = None

    @property
    def scope(self) -> ModuleImportScope:
        """Return the runtime scope governed by a module import block."""

        return ModuleImportScope.RUNTIME

    @property
    def initial_line_number(self) -> int:
        """Return the first import anchor after any module docstring and the
        blank lines that separate it."""

        body = self.module.body
        if not body:
            return 1
        if self._first_statement_index_after_docstring(body):
            docstring = body[0]
            line = (docstring.end_lineno or docstring.lineno) + 1
            return line + DestinationInsertionSpacing.from_source(
                self.source, line, inserted_source_is_import_block=True
            ).following_blank_line_count
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
        guards = TypeCheckingGuardProjection.from_module(self.module).guards
        imports: list[ast.Import | ast.ImportFrom] = []
        for statement in body[self._first_statement_index_after_docstring(body) :]:
            match statement:
                case ast.Import() | ast.ImportFrom():
                    imports.append(statement)
                case ast.If() if any(statement is guard for guard in guards):
                    # A type-checking guard inside the block; its imports go on past it.
                    continue
                case _:
                    break
        return tuple(imports)

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
class TypeCheckingGuardImportInsertionPoint(ImportBlockInsertionPointABC):
    """Insertion geometry derived from one existing type-checking guard."""

    source: str
    guard: ast.If
    file_path: str

    @property
    def scope(self) -> ModuleImportScope:
        """Return the guarded scope governed by this import block."""

        return ModuleImportScope.TYPE_CHECKING

    @property
    def leading_import_statements(self) -> tuple[ImportStatement, ...]:
        imports: list[ImportStatement] = []
        for statement in self.guard.body:
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                break
            imports.append(statement)
        return tuple(imports)

    @property
    def initial_line_number(self) -> int:
        """Return the first statement line inside the type-checking guard."""

        return self.guard.body[0].lineno

    @property
    def indentation(self) -> str:
        return _line_indentation(self.source, self.guard.body[0].lineno)


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
class ModuleImportMutation(KeyedSourceEditCoalescence):
    """Typed additions and removals resolved once across module import scopes."""

    file_path: str
    additions: tuple[RequestedImportStatement, ...] = ()
    removals: tuple[ImportNameRemoval, ...] = ()
    bound_name_removals: tuple[ImportBoundNameRemoval, ...] = ()

    @property
    def coalescence_key(self) -> str:
        return self.file_path

    @classmethod
    def from_source(
        cls,
        *,
        file_path: str,
        import_source: str,
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
        rationale: str = "",
    ) -> ModuleImportMutation:
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
    ) -> ModuleImportMutation:
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
    ) -> ModuleImportMutation:
        """Remove exact import bindings after current-source resolution."""

        return cls(
            file_path=file_path,
            bound_name_removals=tuple(
                ImportBoundNameRemoval(name=name, scope=scope)
                for name in dict.fromkeys(names)
            ),
            rationale=rationale,
        )

    @classmethod
    def _coalesced_group(
        cls,
        mutations: tuple[ModuleImportMutation, ...],
    ) -> ModuleImportMutation:
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
        )

    @staticmethod
    def _coalesced_additions(
        additions: Iterable[RequestedImportStatement],
    ) -> tuple[RequestedImportStatement, ...]:
        aliases_by_family: dict[
            tuple[
                ModuleImportScope,
                ImportSourceGroup,
                type[ImportStatement],
                int,
                str | None,
            ],
            list[ImportAlias],
        ] = {}
        statement_by_family: dict[
            tuple[
                ModuleImportScope,
                ImportSourceGroup,
                type[ImportStatement],
                int,
                str | None,
            ],
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

    @staticmethod
    def _drop_unread_guard_flag(
        module: ast.Module,
        guard_projection: TypeCheckingGuardProjection,
        deletable_guards: frozenset[int],
        import_from_statements: tuple[ast.ImportFrom, ...],
        scope_by_statement_id: dict[int, ModuleImportScope],
        aliases_by_statement: dict[int, list[ImportAlias]],
    ) -> None:
        """This mutation adds `from typing import TYPE_CHECKING` for a guard it
        writes; when the guards it deletes held the module's last reads of that
        flag, the binding goes with them, or the module keeps an import nothing
        reads."""

        deleted_reads = frozenset(
            id(node)
            for guard in guard_projection.guards
            if id(guard) in deletable_guards
            for node in ast.walk(guard.test)
        )
        flag = ImportAlias("TYPE_CHECKING")
        if not deleted_reads or any(
            isinstance(node, ast.Name) and node.id == flag.name and id(node) not in deleted_reads
            for node in ast.walk(module)
        ):
            return
        for statement in import_from_statements:
            if (
                statement.module == "typing"
                and not statement.level
                and scope_by_statement_id[id(statement)] is ModuleImportScope.RUNTIME
            ):
                aliases_by_statement[id(statement)] = [
                    alias for alias in aliases_by_statement[id(statement)] if alias != flag
                ]

    def resolved_windows(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source = context.sources_by_file_path[self.file_path]
        module = context.module_nodes_by_file_path.get(self.file_path)
        if module is None:
            module = ast.parse(source, filename=self.file_path)
        additions = list(
            self._coalesced_additions(
                addition.in_file(self.file_path) for addition in self.additions
            )
        )
        guard_projection = TypeCheckingGuardProjection.from_module(module)
        if (
            any(addition.scope.is_guarded for addition in additions)
            and guard_projection.preferred_reference_source is None
        ):
            additions = list(
                self._coalesced_additions(
                    (
                        *additions,
                        *(
                            addition.in_file(self.file_path)
                            for addition in RequestedImportStatement.from_source(
                                "from typing import TYPE_CHECKING\n"
                            )
                        ),
                    )
                )
            )
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
                ImportAlias.from_alias(alias) for alias in statement.names
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
            retained_aliases: list[ImportAlias] = []
            for alias in aliases_by_statement[id(statement)]:
                removals = requested_bound_name_removals.intersection(
                    ImportBoundNameRemoval(name=bound_name, scope=scope)
                    for bound_name in alias.bound_names_in(
                        requested_statement.declaration
                    )
                )
                if removals:
                    matched_bound_name_removals |= removals
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

        importing_file = context.module_import_graph.source_file_for_path(self.file_path)
        package = (  # a module this stage creates has no digest yet: spellings must agree
            "" if importing_file is None
            else ".".join(importing_file.module_path_identity.package_parts)
        )

        def imported_module(name: ImportFromModuleName) -> str:
            """The module a from-import names, whether spelled absolute or relative."""
            try:
                return name.resolve_from_package(lambda: package)
            except ValueError:
                return name.source

        pending_additions: list[RequestedImportStatement] = []
        for addition in additions:
            matching_from_statements = tuple(
                statement
                for statement in import_from_statements
                if scope_by_statement_id[id(statement)] is addition.scope
                if addition.module_name is not None
                and imported_module(addition.module_name)
                == imported_module(ImportFromModuleName.from_node(statement))
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
            aliases_by_statement[id(target_statement)] = list(
                sorted_tuple(aliases, key=lambda alias: alias.canonical_key)
            )

        replacements: list[PhysicalSourceEdit] = []
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
        self._drop_unread_guard_flag(
            module, guard_projection, deletable_guards, import_from_statements,
            scope_by_statement_id, aliases_by_statement,
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
                span=SourceLineSpan(guard.lineno, guard.end_lineno or guard.lineno),
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
                ImportAlias.from_alias(alias) for alias in statement.names
            )
            aliases = tuple(aliases_by_statement[id(statement)])
            if aliases == original_aliases:
                continue
            if _node_contains_comment(source, statement):
                raise ValueError(
                    f"Cannot rewrite commented import at {self.file_path!r}:"
                    f"{statement.lineno}"
                )
            scope = scope_by_statement_id[id(statement)]
            if not aliases and id(statement) not in preserved_empty_guard_statement_ids:
                replacements.append(
                    SourceSpanDeletion(
                        file_path=self.file_path,
                        span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
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
                .source_in(self.file_path)
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
                    span=SourceLineSpan(statement.lineno, statement.end_lineno or statement.lineno),
                    replacement_lines=SourceTextGeometry(source).generated_lines(
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
            return self._new_guard_insertions(
                source=source,
                module=module,
                runtime_additions=runtime_additions,
                guarded_additions=guarded_additions,
                guard_projection=guard_projection,
            )
        insertions: list[SourceInsertion] = []
        if runtime_additions:
            insertions.extend(
                self._runtime_import_insertions(source, module, runtime_additions)
            )
        if guarded_additions:
            insertions.extend(
                self._existing_guard_insertions(
                    source,
                    guard_projection.guards[0],
                    guarded_additions,
                )
            )
        return tuple(insertions)

    def _runtime_import_insertions(
        self,
        source: str,
        module: ast.Module,
        additions: tuple[RequestedImportStatement, ...],
    ) -> tuple[SourceInsertion, ...]:
        insertion_point = ModuleImportInsertionPoint(
            source,
            self.file_path,
            module_node=module,
        )
        existing_imports = insertion_point.leading_import_statements
        return tuple(
            self._runtime_import_group_insertion(
                source=source,
                insertion_point=insertion_point,
                slot=slot,
                additions=indexed_additions,
                existing_imports=existing_imports,
            )
            for slot, indexed_additions in insertion_point.grouped_additions(additions)
        )

    def _runtime_import_group_insertion(
        self,
        *,
        source: str,
        insertion_point: ModuleImportInsertionPoint,
        slot: ImportInsertionSlot,
        additions: tuple[RequestedImportStatement, ...],
        existing_imports: tuple[ast.Import | ast.ImportFrom, ...],
    ) -> SourceInsertion:
        insertion_line = slot.line_number(insertion_point)
        inserted_source = insertion_point.source_for_insertion(
            source=source,
            slot=slot,
            additions=additions,
        )
        spacing = DestinationInsertionSpacing.from_source(
            source,
            insertion_line,
            inserted_source_is_import_block=True,
        )
        if slot.index == len(existing_imports):
            inserted_source += spacing.trailing_separator
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTextGeometry(source).generated_lines(inserted_source),
            rationale=self.rationale or f"Ensure imports exist in {self.file_path!r}.",
            contributors=self.contributors,
            origins=self.origins,
        )

    def _existing_guard_insertions(
        self,
        source: str,
        guard: ast.If,
        additions: tuple[RequestedImportStatement, ...],
    ) -> tuple[SourceInsertion, ...]:
        insertion_point = TypeCheckingGuardImportInsertionPoint(
            source, guard, self.file_path
        )
        existing_imports = insertion_point.leading_import_statements
        return tuple(
            self._guarded_import_group_insertion(
                source=source,
                insertion_point=insertion_point,
                slot=slot,
                additions=indexed_additions,
                existing_imports=existing_imports,
            )
            for slot, indexed_additions in insertion_point.grouped_additions(additions)
        )

    def _guarded_import_group_insertion(
        self,
        *,
        source: str,
        insertion_point: TypeCheckingGuardImportInsertionPoint,
        slot: ImportInsertionSlot,
        additions: tuple[RequestedImportStatement, ...],
        existing_imports: tuple[ImportStatement, ...],
    ) -> SourceInsertion:
        insertion_line = slot.line_number(insertion_point)
        inserted_source = insertion_point.source_for_insertion(
            source=source,
            slot=slot,
            additions=additions,
        )
        if slot.index == len(existing_imports):
            if len(existing_imports) < len(insertion_point.guard.body):
                inserted_source += "\n"
        return SourceInsertion(
            file_path=self.file_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTextGeometry(source).generated_lines(
                _indent_source(inserted_source, insertion_point.indentation)
            ),
            rationale=self.rationale
            or f"Ensure type-checking imports exist in {self.file_path!r}.",
            contributors=self.contributors,
            origins=self.origins,
        )

    def _new_guard_insertions(
        self,
        *,
        source: str,
        module: ast.Module,
        runtime_additions: tuple[RequestedImportStatement, ...],
        guarded_additions: tuple[RequestedImportStatement, ...],
        guard_projection: TypeCheckingGuardProjection,
    ) -> tuple[SourceInsertion, ...]:
        reference_source = guard_projection.preferred_reference_source
        if reference_source is None:
            reference_source = "TYPE_CHECKING"
        insertion_point = ModuleImportInsertionPoint(
            source,
            self.file_path,
            module_node=module,
        )
        import_boundary_index = len(insertion_point.leading_import_statements)
        boundary_runtime_additions = tuple(
            addition
            for addition in runtime_additions
            if insertion_point.insertion_index_for(addition) == import_boundary_index
        )
        preceding_runtime_additions = tuple(
            addition
            for addition in runtime_additions
            if insertion_point.insertion_index_for(addition) != import_boundary_index
        )
        runtime_source = RequestedImportBlock(
            boundary_runtime_additions, self.file_path
        ).source_after(
            insertion_point.previous_import_statement
        )
        guarded_source = RequestedImportBlock(guarded_additions, self.file_path).source_after(
            None
        )
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
        return (
            *self._runtime_import_insertions(
                source,
                module,
                preceding_runtime_additions,
            ),
            SourceInsertion(
                file_path=self.file_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTextGeometry(source).generated_lines(
                    inserted_source.rstrip("\n") + spacing.trailing_separator
                ),
                rationale=self.rationale
                or f"Ensure type-checking imports exist in {self.file_path!r}.",
                contributors=self.contributors,
                origins=self.origins,
            ),
        )


class ImportSourceGroup(StrEnum):
    """Canonical Python import group derived from an import declaration."""

    FUTURE = ("future", 0, lambda reference, first_party: False)
    STANDARD_LIBRARY = (
        "standard_library",
        1,
        lambda reference, first_party: not reference.is_relative
        and reference.root_module_name in sys.stdlib_module_names
        and reference.root_module_name not in first_party,
    )
    THIRD_PARTY = (
        "third_party",
        2,
        lambda reference, first_party: not reference.is_relative
        and reference.root_module_name not in sys.stdlib_module_names
        and reference.root_module_name not in first_party,
    )
    FIRST_PARTY = (
        "first_party",
        3,
        lambda reference, first_party: not reference.is_relative
        and reference.root_module_name in first_party,
    )
    RELATIVE = ("relative", 4, lambda reference, first_party: reference.is_relative)

    def __new__(
        cls,
        value: str,
        canonical_rank: int,
        reference_matcher: Callable[[ImportFromModuleName, frozenset[str]], bool],
    ) -> ImportSourceGroup:
        member = str.__new__(cls, value)
        member._value_ = value
        member._canonical_rank = canonical_rank
        member._reference_matcher = reference_matcher
        return member

    @property
    def canonical_rank(self) -> int:
        """Return deterministic import-block source order, not dispatch priority."""

        return self._canonical_rank

    @classmethod
    def from_declaration(
        cls, declaration: ImportDeclarationABC, first_party: frozenset[str] = frozenset()
    ) -> tuple[ImportSourceGroup, ...]:
        """`first_party` is the importing repository's own top-level modules
        (`DeclaredRoot.first_party_for`): they sort after third-party imports."""
        if declaration.is_future_import:
            return (cls.FUTURE,)
        return tuple(
            dict.fromkeys(
                cls.from_module_reference(reference, first_party)
                for reference in declaration.module_references
            )
        )

    @classmethod
    def from_module_reference(
        cls, reference: ImportFromModuleName, first_party: frozenset[str] = frozenset()
    ) -> ImportSourceGroup:
        """Require one declaration-owned match, independent of source-order rank."""
        matches = tuple(
            group for group in cls if group._reference_matcher(reference, first_party)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Import reference has no unique canonical source group: {reference.source!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class RequestedImportStatement:
    """AST-normalized import requirement for idempotent import insertion."""

    statement: ast.Import | ast.ImportFrom
    scope: ModuleImportScope = ModuleImportScope.RUNTIME
    first_party: frozenset[str] = frozenset()
    """The importing repository's own top-level modules, once the statement is
    placed in a file (`in_file`); they decide its first-party group."""

    def in_file(self, file_path: str) -> RequestedImportStatement:
        return replace(
            self, first_party=DeclaredRoot.first_party_for(Path(file_path))
        )

    @cached_property
    def declaration(self) -> ImportDeclarationABC:
        return ImportBoundNameProjection(self.statement).declaration

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
    ) -> tuple[RequestedImportStatement, ...]:
        module = ast.parse(source, filename="<requested-import>")
        if not all(
            isinstance(statement, (ast.Import, ast.ImportFrom))
            for statement in module.body
        ):
            return ()
        return tuple(
            requirement
            for statement in module.body
            for requirement in cls.from_statement(statement, scope=scope)
        )

    @classmethod
    def from_statement(
        cls,
        statement: ImportStatement,
        *,
        scope: ModuleImportScope = ModuleImportScope.RUNTIME,
    ) -> tuple[RequestedImportStatement, ...]:
        """Normalize a direct import to one source group per requirement."""

        if isinstance(statement, ast.ImportFrom):
            return (cls(statement, scope=scope),)
        return tuple(
            cls(ast.Import(names=[alias]), scope=scope) for alias in statement.names
        )

    @property
    def aliases(self) -> tuple[ImportAlias, ...]:
        return self.declaration.aliases

    @property
    def module_name(self) -> ImportFromModuleName | None:
        if not isinstance(self.statement, ast.ImportFrom):
            return None
        return ImportFromModuleName.from_node(self.statement)

    @property
    def family_identity(
        self,
    ) -> tuple[
        ModuleImportScope,
        ImportSourceGroup,
        type[ImportStatement],
        int,
        str | None,
    ]:
        if isinstance(self.statement, ast.Import):
            return self.scope, self.source_group, ast.Import, 0, None
        return (
            self.scope,
            self.source_group,
            ast.ImportFrom,
            self.statement.level,
            self.statement.module,
        )

    @property
    def canonical_family_key(self) -> tuple[int, int, str, int, str]:
        """Return deterministic source order without encoding semantic priority."""

        if isinstance(self.statement, ast.Import):
            return (
                self.scope.canonical_rank,
                self.source_group.canonical_rank,
                type(self.statement).__name__,
                0,
                "",
            )
        return (
            self.scope.canonical_rank,
            self.source_group.canonical_rank,
            type(self.statement).__name__,
            self.statement.level,
            self.statement.module or "",
        )

    @property
    def is_future_import(self) -> bool:
        """Return whether this statement belongs to Python's future-import group."""

        return self.declaration.is_future_import

    @property
    def is_relative_import(self) -> bool:
        """Return whether this statement belongs to the relative-import group."""

        return self.source_group is ImportSourceGroup.RELATIVE

    @property
    def source_groups(self) -> tuple[ImportSourceGroup, ...]:
        """Return groups derived from this statement's actual declaration."""
        return ImportSourceGroup.from_declaration(self.declaration, self.first_party)

    @property
    def source_group(self) -> ImportSourceGroup:
        """Return the one canonical group represented by this statement."""

        if len(self.source_groups) != 1:
            raise ValueError(
                "One import statement spans multiple canonical source groups: "
                f"{ast.unparse(self.statement)!r}"
            )
        return self.source_groups[0]

    @property
    def bound_names(self) -> tuple[str, ...]:
        return self.declaration.names()

    def source_in(self, file_path: str) -> str:
        """This import as written into `file_path`, fitted to its line length."""

        return self._rendered(ImportLayout.of(self.statement, file_path))

    @property
    def one_name_source(self) -> str:
        """The source of an import of exactly one name, which every layout
        writes on one line, so no file's line length decides it."""

        if len(self.statement.names) != 1:
            raise ValueError(
                f"An import of {len(self.statement.names)} names needs its file's layout: "
                f"{ast.unparse(self.statement)!r}"
            )
        return self._rendered(OneLineImportLayout())

    def _rendered(self, layout: ImportLayout) -> str:
        if isinstance(self.statement, ast.Import):
            return "".join(
                f"import {ImportFromSource.alias_source(alias)}\n"
                for alias in self.statement.names
            )
        return ImportFromSource(
            module_name=ImportFromModuleName.from_node(self.statement).source,
            aliases=tuple(self.statement.names),
            layout=layout,
        ).source

    def with_aliases(
        self,
        aliases: Iterable[ImportAlias],
    ) -> RequestedImportStatement:
        alias_nodes = [
            alias.ast_alias for alias in aliases
        ]
        # The merged statement takes the original's location, so its layout is
        # the one the original was written in.
        if isinstance(self.statement, ast.Import):
            return RequestedImportStatement(
                ast.copy_location(ast.Import(names=alias_nodes), self.statement),
                scope=self.scope,
                first_party=self.first_party,
            )
        return RequestedImportStatement(
            ast.copy_location(
                ast.ImportFrom(
                    module=self.statement.module,
                    names=alias_nodes,
                    level=self.statement.level,
                ),
                self.statement,
            ),
            scope=self.scope,
            first_party=self.first_party,
        )


@dataclass(frozen=True)
class RequestedImportBlock:
    """Canonical source for a group of requested import statements."""

    statements: tuple[RequestedImportStatement, ...]
    file_path: str

    def source_after(
        self,
        previous_statement: ast.Import | ast.ImportFrom | None,
    ) -> str:
        previous = (
            RequestedImportStatement(previous_statement).in_file(self.file_path)
            if previous_statement is not None
            else None
        )
        source_parts: list[str] = []
        for statement in (statement.in_file(self.file_path) for statement in self.statements):
            if (
                previous is not None
                and previous.source_group is not statement.source_group
            ):
                source_parts.append("\n")
            source_parts.append(statement.source_in(self.file_path))
            previous = statement
        return "".join(source_parts)


class ImportLayout(ABC):
    """How a rendered from-import is laid out. A rewrite keeps the layout of the
    import it rewrites, so a move changes names, not shapes."""

    @abstractmethod
    def render(self, head: str, alias_sources: tuple[str, ...]) -> str:
        """The import `head` (``from x import ``) of these aliases."""

    @staticmethod
    def one_line(head: str, alias_sources: tuple[str, ...]) -> str:
        return f"{head}{', '.join(alias_sources)}\n"

    @staticmethod
    def column(head: str, alias_sources: tuple[str, ...]) -> str:
        alias_lines = "".join(f"    {alias_source},\n" for alias_source in alias_sources)
        return f"{head}(\n{alias_lines})\n"

    @classmethod
    def of(cls, statement: ast.Import | ast.ImportFrom, file_path: str) -> ImportLayout:
        """The layout of `statement` as written in `file_path`: a parenthesised
        import stays a column, a one-line import stays one line while it fits."""

        match statement:
            case ast.ImportFrom(lineno=int(start), end_lineno=int(end)) if end != start:
                return ColumnImportLayout()
            case ast.ImportFrom(col_offset=int(indent)):
                return FittedImportLayout(
                    DeclaredRoot.line_length_for(Path(file_path)) - indent
                )
        # Built in code, not written anywhere: a new import, fitted to the file.
        return cls.fitted(file_path)

    @classmethod
    def fitted(cls, file_path: str) -> ImportLayout:
        """A new import in `file_path`: one line while it fits the file's limit."""

        return FittedImportLayout(DeclaredRoot.line_length_for(Path(file_path)))

    @classmethod
    def in_string(cls) -> ImportLayout:
        """An import written inside a string literal, which keeps it on one line."""

        return OneLineImportLayout()


class OneLineImportLayout(ImportLayout):
    def render(self, head: str, alias_sources: tuple[str, ...]) -> str:
        return self.one_line(head, alias_sources)


@dataclass(frozen=True)
class FittedImportLayout(ImportLayout):
    line_length: int

    def render(self, head: str, alias_sources: tuple[str, ...]) -> str:
        line = self.one_line(head, alias_sources)
        if len(alias_sources) == 1 or len(line) - 1 <= self.line_length:
            return line
        return self.column(head, alias_sources)


class ColumnImportLayout(ImportLayout):
    def render(self, head: str, alias_sources: tuple[str, ...]) -> str:
        if len(alias_sources) == 1:
            return self.one_line(head, alias_sources)
        return self.column(head, alias_sources)


@dataclass(frozen=True)
class ImportFromSource:
    """Rendered from-import source for remaining aliases."""

    module_name: str
    aliases: tuple[ast.alias, ...]
    layout: ImportLayout

    @property
    def source(self) -> str:
        if not self.aliases:
            return ""
        return self.layout.render(f"from {self.module_name} import ", self.alias_sources)

    @property
    def alias_sources(self) -> tuple[str, ...]:
        return tuple(self.alias_source(alias) for alias in self.aliases)

    @staticmethod
    def alias_source(alias: ast.alias) -> str:
        return ImportAlias.from_alias(alias).source
