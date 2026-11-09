from __future__ import annotations

import ast
import builtins
import os
import re
import tomllib
from abc import (
    ABC,
    abstractmethod,
)
from collections import Counter
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePosixPath
from typing import ClassVar, cast

from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.declared_family import ForeignAdapter
from nominal_refactor_advisor.source_identity import PACKAGE_INIT_FILE_NAME
from nominal_refactor_advisor.ast_tools import (
    ModuleAnnotationEvaluationMode,
    ParsedModule,
    ParsedModuleFact,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.codemod_import_bindings import (
    FromModuleImportBindingIdentity,
)
from nominal_refactor_advisor.codemod_import_graph import SourceModuleImportGraph
from nominal_refactor_advisor.source_index import SourceFileDigest
from nominal_refactor_advisor.codemod_import_scopes import (
    ModuleImportScope,
    TypeCheckingGuardProjection,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportFromSource,
    ImportLayout,
    ModuleImportMutation,
)
from nominal_refactor_advisor.codemod_module_declarations import (
    ModuleSymbolTable,
    MovedTopLevelDeclarationSource,
    SourceTopLevelSymbolMoveSelection,
)
from nominal_refactor_advisor.codemod_module_move_reports import (
    ModuleMoveDependencyReport,
    ModuleMoveImportDependency,
    ModuleMoveObstacle,
    ModuleMoveObstacleKind,
    ModuleMoveSourceLocalDependency,
    ModuleMoveSourceLocalDependencyResolution,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_paths import SourcePathResolutionAuthority
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourcePathPayloadValueCodec
from nominal_refactor_advisor.codemod_semantics import (
    CodemodPreflightStatus,
    CodemodSourceDependencyScope,
)
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceFileDeletion,
    SourceInsertion,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
    TextFileCarryEnd,
    TextFileChange,
    TextFileEdit,
)
from nominal_refactor_advisor.codemod_spacing import DestinationInsertionSpacing
from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyProjection,
)
from nominal_refactor_advisor.json_reports import DataclassJsonReport
from nominal_refactor_advisor.lexical_bindings import (
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ImportBoundNameProjection,
    ImportFromModuleName,
)
from nominal_refactor_advisor.public_exports import (
    CompactModulePublicExportContract,
    module_public_export_contract,
)
from nominal_refactor_advisor.python_module_identity import PythonModulePathIdentity
from nominal_refactor_advisor.value_expression import LexicalValueReference


@dataclass(frozen=True, kw_only=True)
class PromoteGuardedImportOperation(RefactorRecipeOperation):
    """Move names out of an ``if TYPE_CHECKING:`` guard into a runtime import.

    A guard that hides no cycle restates nothing but a fear of one; this proves
    against the repository's runtime import graph that the import closes no cycle,
    and refuses (naming the module that reaches back) when it would."""

    module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    import_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
    )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        import_graph = context.module_import_graph
        source_file = import_graph.source_file_for_path(source_path)
        guarded = frozenset(
            alias.name
            for statement in ModuleImportScope.TYPE_CHECKING.import_statements(
                context.module_nodes_by_file_path[source_path]
            )
            if isinstance(statement, ast.ImportFrom)
            and ImportFromModuleName.from_node(statement).source == self.module_name
            for alias in statement.names
        )
        missing = tuple(name for name in self.import_names if name not in guarded)
        imported_module = source_file.module_path_identity.resolve_import_from_module(
            imported_module=self.module_name.lstrip("."),
            level=len(self.module_name) - len(self.module_name.lstrip(".")),
        )
        imported_file = (
            None
            if imported_module is None
            else import_graph.unique_source_file_for_module_name(imported_module)
        )
        cycle = imported_file is not None and import_graph.import_would_create_cycle(
            importing_file_path=source_path,
            imported_file_path=imported_file.file_path,
        )
        message = (
            f"{self.import_names!r} are not guarded imports of {self.module_name!r}: "
            f"{missing!r}"
            if missing
            else f"{imported_module!r} already imports {source_file.module_name!r} at "
            "runtime; the guard hides a cycle, so move the declaration instead"
            if cycle
            else "The guarded import closes no runtime import cycle"
        )
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=(
                    CodemodPreflightStatus.FAILED
                    if missing or cycle
                    else CodemodPreflightStatus.PASSED
                ),
                message=message,
                detail=PromotedGuardedImportReport(
                    file_path=source_path,
                    module_name=self.module_name,
                    import_names=self.import_names,
                ),
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        rationale = self.rationale or (
            f"Import {self.import_names!r} from {self.module_name!r} at runtime."
        )
        return (
            ModuleImportMutation.remove_names(
                file_path=source_path,
                module_name=self.module_name,
                names=self.import_names,
                scope=ModuleImportScope.TYPE_CHECKING,
                rationale=rationale,
            ),
            ModuleImportMutation.from_source(
                file_path=source_path,
                import_source=(
                    f"from {self.module_name} import {', '.join(self.import_names)}\n"
                ),
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class PromotedGuardedImportReport(DataclassJsonReport):
    """Which guarded names a promotion moves to runtime."""

    file_path: str
    module_name: str
    import_names: tuple[str, ...]


@dataclass(frozen=True)
class FilesReachingModule:
    """The parsed files that can reach one module, for every pass over its consumers.

    A file that imports, binds, reads or names `pkg.source` (its dotted name, its file
    name, an import written in a string) spells `source` in its text. Inside a package
    whose `__init__.py` is the module, `from . import name` reaches it without spelling
    it, so those files count too. Every other file is skipped unread: on agent-comms with
    its tests, a dissolve otherwise walked about 1,100 modules to find the few dozen that
    name it. Each file comes with its parse, whose `ModuleNodeIndex` fact every pass shares."""

    context: CodemodSelectorContext
    module_name: str

    def __iter__(self) -> Iterator[tuple[SourceFileDigest, ParsedModule]]:
        leaf = self.module_name.rpartition(".")[2]
        owner = self.context.module_import_graph.unique_source_file_for_module_name(
            self.module_name
        )
        package_files = tuple(
            owner.file_path.removesuffix("__init__.py")
            for owner in (owner,)
            if owner is not None and owner.file_path.endswith("/__init__.py")
        )
        for source_file in self.context.source_index.files:
            if source_file.file_path in self.context.module_nodes_by_file_path and (
                leaf in self.context.sources_by_file_path[source_file.file_path]
                or source_file.file_path.startswith(package_files)
            ):
                yield source_file, self.context.parsed_module_for_source_path(
                    source_file.file_path
                )


@dataclass(frozen=True)
class ModuleNameRedirect:
    """Every importer of names leaving one module, redirected to each name's new
    owner, so each importer keeps working:

    - `from source import Name` imports the name from its owner;
    - `from source import *` gains each moved name the source's export contract
      proves public, as `from owner import Name as Name`, so it stays exposed
      through the star;
    - a `<module>.Name` read through a binding of the source module object reads
      `Name`, imported from its owner.

    A reader it cannot rewrite is refused (`_SourceModuleObjectReads`,
    `_StarExportedNames`)."""

    source_path: str
    destinations: Mapping[str, str]

    def consumer_edits(
        self,
        context: CodemodSelectorContext,
        *,
        excluded_paths: frozenset[str],
    ) -> tuple[NominalSourceEdit, ...]:
        import_graph = context.module_import_graph
        source_module_name = self.source_module_name(import_graph)
        moved_names = frozenset(self.destinations)
        star_exported = _StarExportedNames.of(
            module_public_export_contract(
                context.parsed_module_for_source_path(self.source_path)
            ),
            tuple(sorted(moved_names)),
        )
        edits: list[NominalSourceEdit] = []
        for source_file, parsed in FilesReachingModule(context, source_module_name):
            consumer_path = source_file.file_path
            if consumer_path in excluded_paths:
                continue
            module = parsed.module
            identity = source_file.module_path_identity
            for scope in ModuleImportScope:
                for statement in scope.import_statements(module):
                    if not isinstance(statement, ast.ImportFrom) or (
                        identity.resolve_import_from_module(
                            imported_module=statement.module,
                            level=statement.level,
                        )
                        != source_module_name
                    ):
                        continue
                    edits.extend(
                        self._owner_imports(
                            import_graph,
                            consumer_path,
                            scope,
                            tuple(
                                ast.alias(name=name, asname=name)
                                for alias in statement.names
                                if alias.name == "*"
                                for name in star_exported.required(consumer_path)
                            ),
                        )
                    )
            retargets = _SourceModuleAliasRetargets.of(
                parsed.fact(ModuleNodeIndex),
                source_module_name=source_module_name,
                owners=_RedirectOwners(self, import_graph),
            )
            if retargets.aliases:
                edits.append(retargets.rewrite(consumer_path, context.sources_by_file_path[consumer_path]))
            module_reads = _SourceModuleObjectReads.of(
                parsed,
                identity,
                source_module_name=source_module_name,
                moved_names=moved_names,
                file_path=consumer_path,
                retargeted=retargets.bindings,
            )
            if module_reads.names:
                edits.extend(
                    self._owner_imports(
                        import_graph,
                        consumer_path,
                        ModuleImportScope.RUNTIME,
                        tuple(ast.alias(name=name) for name in module_reads.names),
                    )
                )
            module_objects = self._accessed_module_objects(
                import_graph, consumer_path, module_reads
            )
            edits.extend(
                ModuleImportMutation.from_source(
                    file_path=consumer_path, import_source=import_source
                )
                for import_source in sorted({source for _node, _name, source in module_objects})
            )
            if module_reads.reads or module_objects:
                edits.append(
                    module_reads.rewrite(
                        context.sources_by_file_path[consumer_path],
                        tuple((node, name) for node, name, _source in module_objects),
                    )
                )
            edits.extend(
                ModuleImportMutation.remove_bound_names(
                    file_path=consumer_path, names=(binding,), scope=scope
                )
                for binding, scope in module_reads.unread_bindings
            )
            run_only = _RunOnlyImports.of(parsed, consumer_path, source_module_name)
            if run_only.statements:
                edits.extend(
                    self.run_import_edits(
                        import_graph, run_only, source_module_name, context.sources_by_file_path[consumer_path]
                    )
                )
            local_imports = self._in_place_import_replacements(
                import_graph,
                parsed,
                identity,
                consumer_path,
                source_module_name,
            )
            edits.extend(self.import_string_edits(context, consumer_path))
            if local_imports:
                geometry = SourceTextGeometry(context.sources_by_file_path[consumer_path])
                edits.append(
                    geometry.nominal_edit(
                        file_path=consumer_path,
                        replacements=tuple(
                            SourceTextSpanReplacement.from_offsets(
                                start_offset=start,
                                end_offset=end,
                                replacement_source=replacement,
                            )
                            for statement, replacement in local_imports
                            for start, end in (geometry.required_node_offsets(statement),)
                        ),
                    )
                )
        return tuple(edits)

    def run_import_edits(
        self,
        import_graph: SourceModuleImportGraph,
        run_only: _RunOnlyImports,
        source_module_name: str,
        source: str,
    ) -> tuple[NominalSourceEdit, ...]:
        """How an import taken only to run the source follows the names leaving
        it. Into their owners, nothing here: a dissolve replaces such an import
        with what the dissolved module still runs (`DissolveModuleOperation`)."""
        del import_graph, run_only, source_module_name, source
        return ()

    def _in_place_import_replacements(
        self,
        import_graph: SourceModuleImportGraph,
        parsed: ParsedModule,
        identity: PythonModulePathIdentity,
        consumer_path: str,
        source_module_name: str,
    ) -> tuple[tuple[ast.ImportFrom, str], ...]:
        """Every `from source import ...` naming moved names, at module level, under
        `TYPE_CHECKING` or inside a body, rewritten where it stands: one statement
        per owner, plus one for the names that stay. The reader's imports keep
        their place and grouping."""

        scope_by_statement = {
            id(statement): scope
            for scope in ModuleImportScope
            for statement in scope.import_statements(parsed.module)
        }
        replacements: list[tuple[ast.ImportFrom, str]] = []
        for statement in parsed.fact(ModuleNodeIndex).import_froms():
            if (
                identity.resolve_import_from_module(
                    imported_module=statement.module,
                    level=statement.level,
                )
                != source_module_name
            ):
                continue
            moved = tuple(alias for alias in statement.names if alias.name in self.destinations)
            if not moved:
                continue
            if any(alias.name == "*" for alias in statement.names):
                raise ValueError(f"{consumer_path!r}:{statement.lineno} star-imports locally")
            kept = tuple(alias for alias in statement.names if alias.name not in self.destinations)
            scope = scope_by_statement.get(id(statement), ModuleImportScope.RUNTIME)
            lines = [
                ImportFromSource(
                    module_name=scope.required_module_reference(
                        import_graph,
                        importing_file_path=consumer_path,
                        imported_file_path=destination_path,
                        imported_name=owned[0].name,
                    ),
                    aliases=owned,
                    layout=ImportLayout.of(statement, consumer_path),
                ).source.strip()
                for destination_path in sorted({self.destinations[alias.name] for alias in moved})
                for owned in (
                    tuple(alias for alias in moved if self.destinations[alias.name] == destination_path),
                )
            ]
            if kept:
                lines.append(
                    ImportFromSource(
                        module_name=ImportFromModuleName.from_node(statement).source,
                        aliases=kept,
                        layout=ImportLayout.of(statement, consumer_path),
                    ).source.strip()
                )
            replacements.append(
                # Each owner's statement can be several lines (a parenthesised
                # list); every line after the first takes the reader's indent.
                (statement, "\n".join(lines).replace("\n", "\n" + " " * statement.col_offset))
            )
        return tuple(replacements)

    def _import_string_replacements(
        self,
        import_graph: SourceModuleImportGraph,
        parsed: ParsedModule,
        source_module_name: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """An import of moved names written as text is a reader: a string that
        is exactly one `from <source> import ...` statement (a plan's
        `import_source`, say) names each owner instead, and so does such a
        statement inside a longer string (a program a test hands a subprocess),
        one `from` per owner joined by `; ` so the text keeps its lines. Names
        that did not move stay on the source. A string that only mentions the
        module is prose, and a dissolve's mention rule owns it."""

        source = parsed.source
        geometry = SourceTextGeometry(source)
        embedded = _EmbeddedImportFrom(source_module_name)
        replacements: list[SourceTextSpanReplacement] = []
        for node in parsed.fact(ModuleNodeIndex).string_constants:
            statement = _SingleImportFromText.of(node.value)
            if statement is None or statement.level or statement.module != source_module_name:
                replacements.extend(
                    self._embedded_import_replacements(import_graph, geometry, node, embedded)
                )
                continue
            moved = tuple(alias for alias in statement.names if alias.name in self.destinations)
            if not moved:
                continue
            kept = tuple(alias for alias in statement.names if alias.name not in self.destinations)
            lines = [
                ImportFromSource(
                    module_name=import_graph.module_name_for_file_path(destination_path),
                    aliases=tuple(
                        alias for alias in moved if self.destinations[alias.name] == destination_path
                    ),
                    layout=ImportLayout.in_string(),
                ).source.strip()
                for destination_path in sorted({self.destinations[alias.name] for alias in moved})
            ]
            if kept:
                lines.append(
                    ImportFromSource(
                        module_name=source_module_name,
                        aliases=kept,
                        layout=ImportLayout.in_string(),
                    ).source.strip()
                )
            start, end = geometry.required_node_offsets(node)
            quote = source[start]
            text = "\n".join(lines)
            if quote not in "\"'" or quote in text or "\n" in text:
                text_literal = repr(text)
            else:
                text_literal = f"{quote}{text}{quote}"
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=end,
                    replacement_source=text_literal,
                )
            )
        return tuple(replacements)

    def source_module_name(self, import_graph: SourceModuleImportGraph) -> str:
        """The dotted name every rewrite of the source's readers matches. A move
        plan has resolved it before any reader is read, so a source without one
        is refused, never read as naming nothing."""

        source_module_name = import_graph.module_name_for_file_path(self.source_path)
        if source_module_name is None:
            raise ValueError(f"Source module identity is unavailable for {self.source_path!r}")
        return source_module_name

    def import_string_edits(
        self,
        context: CodemodSelectorContext,
        file_path: str,
    ) -> tuple[NominalSourceEdit, ...]:
        """The edit that points one file's import strings at each moved name's owner."""

        import_graph = context.module_import_graph
        replacements = self._import_string_replacements(
            import_graph,
            context.parsed_module_for_source_path(file_path),
            self.source_module_name(import_graph),
        )
        if not replacements:
            return ()
        return (
            SourceTextGeometry(context.sources_by_file_path[file_path]).nominal_edit(
                file_path=file_path,
                replacements=replacements,
            ),
        )

    def moved_text(self, import_graph: SourceModuleImportGraph, text: str) -> str:
        """Code leaving the source reads the source's names as text too: an import
        string in the moved code follows each moved name to its owner, the way a
        consumer's does, so the moved text never names a module that no longer
        binds them."""

        source_module_name = self.source_module_name(import_graph)
        parsed = ParsedModule(
            path=Path(self.source_path),
            module_name=source_module_name,
            module=ast.parse(text),
            source=text,
        )
        return TextFileEdit.applied(
            text,
            sorted(
                (replacement.start_offset, replacement.end_offset, replacement.replacement_source)
                for replacement in self._import_string_replacements(
                    import_graph, parsed, source_module_name
                )
            ),
        )

    def _embedded_import_replacements(
        self,
        import_graph: SourceModuleImportGraph,
        geometry: SourceTextGeometry,
        node: ast.Constant,
        embedded: _EmbeddedImportFrom,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        offsets = geometry.node_offsets(node)
        if offsets is None:
            return ()
        start, end = offsets
        replacements: list[SourceTextSpanReplacement] = []
        for match in embedded.matches(geometry.source[start:end]):
            names = embedded.names(match)
            if not any(name in self.destinations for name in names):
                continue
            by_owner: dict[str, list[str]] = {}
            for name in names:
                owner = (
                    import_graph.module_name_for_file_path(self.destinations[name])
                    if name in self.destinations
                    else embedded.module_name
                )
                by_owner.setdefault(owner, []).append(name)
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start + match.start(),
                    end_offset=start + match.end(),
                    replacement_source="; ".join(
                        f"from {owner} import {', '.join(owned)}"
                        for owner, owned in by_owner.items()
                    ),
                )
            )
        return tuple(replacements)

    def name_reader_paths(self, name: str) -> frozenset[str]:
        """The modules whose code reads `name` once it leaves the source. These
        owners spread the readers, so no module is known to be one."""

        del name
        return frozenset()

    def binding_reader_paths(self, name: str) -> frozenset[str]:
        """The modules that bind `name` once the move lands besides the name's own
        owner: an import the moved code carries, or a module-context name
        (`__name__`, say) that follows the code. These owners create
        no module, so there is none."""

        del name
        return frozenset()

    def patched_reader_path(self, name: str, file_path: str) -> str:
        """The module a patch of `source.name` must target instead: its one reader."""

        match tuple(self.name_reader_paths(name)):
            case (reader_path,):
                return reader_path
            case _:
                raise ValueError(
                    f"{file_path!r} patches {name!r} on the source module, whose readers "
                    "the move spreads across owners"
                )

    def _accessed_module_objects(
        self,
        import_graph: SourceModuleImportGraph,
        consumer_path: str,
        module_reads: _SourceModuleObjectReads,
    ) -> tuple[tuple[ast.expr, str, str], ...]:
        """Each by-name access's module object, the binding of the module it now
        targets, and the import that binds it."""

        module_objects: list[tuple[ast.expr, str, str]] = []
        for node, access in module_reads.accesses:
            module_name = import_graph.module_name_for_file_path(
                access.target_path(self, consumer_path)
            )
            if module_name is None:
                raise ValueError(
                    f"{consumer_path!r} reaches {access.member_name!r} through "
                    f"{access.accessor!r} in a module outside the index"
                )
            package, _, binding = module_name.rpartition(".")
            if binding in module_reads.bound_names:
                raise ValueError(
                    f"{consumer_path!r} already binds {binding!r}, which its "
                    f"{access.accessor!r} of {access.member_name!r} would be "
                    "redirected to"
                )
            module_objects.append(
                (
                    node,
                    binding,
                    f"from {package} import {binding}\n" if package else f"import {binding}\n",
                )
            )
        return tuple(module_objects)

    def _owner_imports(
        self,
        import_graph: SourceModuleImportGraph,
        consumer_path: str,
        scope: ModuleImportScope,
        aliases: tuple[ast.alias, ...],
    ) -> Iterator[ModuleImportMutation]:
        """One import per owner, of the aliases that owner now declares."""

        for destination_path in sorted({self.destinations[alias.name] for alias in aliases}):
            owned = tuple(
                alias for alias in aliases if self.destinations[alias.name] == destination_path
            )
            yield ModuleImportMutation.from_source(
                file_path=consumer_path,
                import_source=ImportFromSource(
                    module_name=scope.required_module_reference(
                        import_graph,
                        importing_file_path=consumer_path,
                        imported_file_path=destination_path,
                        imported_name=owned[0].name,
                    ),
                    aliases=owned,
                    layout=ImportLayout.fitted(consumer_path),
                ).source,
                scope=scope,
            )


@dataclass(frozen=True)
class RelocationNameRedirect(ModuleNameRedirect):
    """The redirect of one relocation: every moved declaration, and so every
    reader of a removed source import, lands in `reader_path`."""

    reader_path: str
    reader_bindings: frozenset[str] = frozenset()
    """The names `reader_path` binds once the move lands besides the moved
    declarations: the imports the moved code carries, and the module-context
    names that follow the code when every name the source declares moves into
    the module this relocation creates."""

    def name_reader_paths(self, name: str) -> frozenset[str]:
        del name
        return frozenset((self.reader_path,))

    def binding_reader_paths(self, name: str) -> frozenset[str]:
        return frozenset((self.reader_path,)) if name in self.reader_bindings else frozenset()

    def run_import_edits(
        self,
        import_graph: SourceModuleImportGraph,
        run_only: _RunOnlyImports,
        source_module_name: str,
        source: str,
    ) -> tuple[NominalSourceEdit, ...]:
        """The moved declarations run in `reader_path` now, so an import taken only
        to run the source also runs it, right after the source."""
        reader = import_graph.module_name_for_file_path(self.reader_path)
        if reader is None:
            raise ValueError(f"Destination module identity is unavailable for {self.reader_path!r}")
        if reader.partition(".")[0] != source_module_name.partition(".")[0]:
            raise ValueError(
                f"{run_only.file_path!r}:{run_only.statements[0].lineno} imports "
                f"{source_module_name!r} to run it, and {reader!r} binds another package head"
            )
        return (run_only.edit(source, source_module_name, (source_module_name, reader)),)


@dataclass(frozen=True)
class _RunOnlyImports:
    """The bare `import pkg.module` statements of one file taken only to run the
    module: nothing in the file reads through the dotted path, so what the import
    does is run the module, and the import follows whatever runs it now."""

    file_path: str
    statements: tuple[ast.Import, ...]

    @classmethod
    def of(cls, parsed: ParsedModule, file_path: str, module_name: str) -> _RunOnlyImports:
        if cls.reads_path(parsed.module, module_name):
            return cls(file_path, ())
        return cls(
            file_path,
            tuple(
                statement
                for statement in parsed.fact(ModuleNodeIndex).imports
                if isinstance(statement, ast.Import)
                and any(alias.asname is None and alias.name == module_name for alias in statement.names)
            ),
        )

    @staticmethod
    def reads_path(module: ast.Module, module_name: str) -> bool:
        parts = tuple(module_name.split("."))
        return any(
            (reference := LexicalValueReference.from_expression(node)) is not None
            and reference.parts == parts
            for node in ast.walk(module)
            if isinstance(node, ast.Attribute)
        )

    def edit(self, source: str, module_name: str, runs: tuple[str, ...]) -> NominalSourceEdit:
        """Each statement imports `runs`, in order, where it imported `module_name`."""
        geometry = SourceTextGeometry(source)
        replacements: list[SourceTextSpanReplacement] = []
        for statement in self.statements:
            names = tuple(
                dict.fromkeys(
                    name
                    for alias in statement.names
                    for name in (
                        runs
                        if alias.asname is None and alias.name == module_name
                        else (ast.unparse(alias),)
                    )
                )
            )
            if not names:
                raise ValueError(
                    f"{self.file_path!r}:{statement.lineno} imports {module_name!r} "
                    "only to run it, and it runs nothing now"
                )
            start, end = geometry.required_node_offsets(statement)
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start, end_offset=end, replacement_source=f"import {', '.join(names)}"
                )
            )
        return geometry.nominal_edit(file_path=self.file_path, replacements=tuple(replacements))


@dataclass(frozen=True)
class _StarExportedNames:
    """The moved names a star import of the source module carries, from the
    source's export contract; a name the contract cannot decide is refused."""

    names: tuple[str, ...]
    unresolved: tuple[str, ...]

    @classmethod
    def of(
        cls,
        contract: CompactModulePublicExportContract,
        moved_symbol_names: tuple[str, ...],
    ) -> _StarExportedNames:
        exposures = tuple((name, contract.exposure_for(name)) for name in moved_symbol_names)
        return cls(
            names=tuple(name for name, exposure in exposures if exposure.proves_public_exposure),
            unresolved=tuple(
                name for name, exposure in exposures if exposure.introduces_uncertainty
            ),
        )

    def required(self, file_path: str) -> tuple[str, ...]:
        if self.unresolved:
            raise ValueError(
                f"{file_path!r} star-imports the source, whose export contract does "
                f"not decide {self.unresolved!r}"
            )
        return self.names


@dataclass(frozen=True)
class _EmbeddedImportFrom:
    """`from <module> import a, b` written anywhere inside a string's source
    text: the one spelling both the redirect (which rewrites it to each name's
    owner) and a dissolve's mention rule (which leaves it to the redirect) read."""

    module_name: str

    def matches(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(
            rf"from[ \t]+{re.escape(self.module_name)}[ \t]+import[ \t]+"
            r"(\w+(?:[ \t]*,[ \t]*\w+)*)",
            text,
        )

    @staticmethod
    def names(match: re.Match[str]) -> list[str]:
        return [name.strip() for name in match.group(1).split(",")]


class _SingleImportFromText:
    """The `from ... import ...` statement a string's whole text is, if it is one."""

    @staticmethod
    def of(text: str) -> ast.ImportFrom | None:
        if "import" not in text:
            return None
        try:
            parsed = ast.parse(text)
        except SyntaxError:
            return None
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.ImportFrom):
            return None
        return parsed.body[0]


@dataclass(frozen=True)
class _NamedModuleMemberAccess:
    """`getattr(m, "Name")`-shaped access: the module object is the first argument
    of a by-name accessor and the member is a string literal, so the name it
    reaches is as static as `m.Name`. Each member says which module the access
    targets once the name leaves the source."""

    accessor: str
    member_name: str

    ACCESSORS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def of(
        cls,
        module_object: ast.expr,
        parents: dict[ast.AST, ast.AST],
    ) -> _NamedModuleMemberAccess | None:
        call = parents.get(module_object)
        if not (
            isinstance(call, ast.Call)
            and len(call.args) >= 2
            and call.args[0] is module_object
            and isinstance(member := call.args[1], ast.Constant)
            and isinstance(member.value, str)
        ):
            return None
        accessor = ast.unparse(call.func)
        for member_type in cls.__subclasses__():
            if accessor in member_type.ACCESSORS:
                return member_type(accessor=accessor, member_name=member.value)
        return None

    def target_path(self, redirect: ModuleNameRedirect, file_path: str) -> str:
        raise NotImplementedError

    def reached_paths(self, redirect: ModuleNameRedirect) -> frozenset[str]:
        """The modules this access may reach once its name leaves the source."""
        raise NotImplementedError


@dataclass(frozen=True)
class _NamedModuleMemberRead(_NamedModuleMemberAccess):
    """`getattr`/`hasattr` reads the name where it is now declared."""

    ACCESSORS: ClassVar[frozenset[str]] = frozenset(
        accessor.__name__ for accessor in (getattr, hasattr)
    )

    def target_path(self, redirect: ModuleNameRedirect, file_path: str) -> str:
        return redirect.destinations[self.member_name]

    def reached_paths(self, redirect: ModuleNameRedirect) -> frozenset[str]:
        return frozenset((redirect.destinations[self.member_name],))


@dataclass(frozen=True)
class _NamedModuleMemberPatch(_NamedModuleMemberAccess):
    """`setattr`/`delattr` (or the `monkeypatch` fixture's) replace the name in the
    module whose code reads it, which is not its declaring module when the name
    is an import: the patch follows the reader."""

    ACCESSORS: ClassVar[frozenset[str]] = frozenset(
        f"{fixture}{accessor.__name__}"
        for fixture, accessor in product(("", "monkeypatch."), (setattr, delattr))
    )

    def target_path(self, redirect: ModuleNameRedirect, file_path: str) -> str:
        return redirect.patched_reader_path(self.member_name, file_path)

    def reached_paths(self, redirect: ModuleNameRedirect) -> frozenset[str]:
        return redirect.name_reader_paths(self.member_name)


class _AliasOwners(ABC):
    """Which modules a use of a source-module alias reaches once the moved names
    leave the source: an attribute read `s.Name`, or a by-name access
    `getattr(s, "Name")`. A use reaching anything but one module keeps the alias."""

    @abstractmethod
    def attribute_owners(self, name: str) -> frozenset[str]:
        raise NotImplementedError

    @abstractmethod
    def access_owners(self, access: _NamedModuleMemberAccess) -> frozenset[str]:
        raise NotImplementedError

    def attribute_readers(self, name: str) -> frozenset[str]:
        """Every module `s.name` can be read from once the move lands; a use
        whose owners name no one module for the alias can still share one of
        these."""

        return self.attribute_owners(name)


@dataclass(frozen=True)
class _RedirectOwners(_AliasOwners):
    """A consumer's alias, followed through the move's redirect."""

    redirect: ModuleNameRedirect
    import_graph: SourceModuleImportGraph

    def attribute_owners(self, name: str) -> frozenset[str]:
        if name not in self.redirect.destinations:
            return frozenset()
        return self._modules((self.redirect.destinations[name],))

    def attribute_readers(self, name: str) -> frozenset[str]:
        """Its owner, and the module a whole moved module's code carries it into."""

        return self.attribute_owners(name) | self._modules(self.redirect.binding_reader_paths(name))

    def access_owners(self, access: _NamedModuleMemberAccess) -> frozenset[str]:
        if access.member_name not in self.redirect.destinations:
            return frozenset()
        return self._modules(access.reached_paths(self.redirect))

    def _modules(self, paths: Iterable[str]) -> frozenset[str]:
        """The modules at `paths`; a path outside the index reaches no module the
        move can name, so the use reaches none and the alias keeps its refusal."""
        modules = frozenset(self.import_graph.module_name_for_file_path(path) for path in paths)
        if None in modules:
            return frozenset()
        return cast(frozenset[str], modules)


@dataclass(frozen=True)
class _DestinationOwners(_AliasOwners):
    """The destination's own alias of the source: its reads of the moved names
    now read the destination itself. A by-name access stays an alias use."""

    module_name: str
    names: frozenset[str]

    def attribute_owners(self, name: str) -> frozenset[str]:
        return frozenset((self.module_name,)) if name in self.names else frozenset()

    def access_owners(self, access: _NamedModuleMemberAccess) -> frozenset[str]:
        del access
        return frozenset()


@dataclass(frozen=True)
class _SourceModuleAliasRetargets:
    """`import source as s` whose every load of `s` reaches a name in one module
    binds that module where it stands: the same scope, the same alias, the uses
    as written. A load reaches its name's owner by `s.Name`, and the module a
    by-name access targets by `getattr(s, "Name")` or `monkeypatch.setattr(s,
    "Name", ...)` (the reader of the name, for a patch). A function that imports
    the module to keep it off the import path at load time keeps doing so. An
    alias that also reads a name staying behind, reaches several modules, or is
    used any other way stays with `_SourceModuleObjectReads`."""

    aliases: tuple[tuple[ast.alias, str], ...]

    @property
    def bindings(self) -> frozenset[str]:
        return frozenset(alias.asname for alias, _owner in self.aliases if alias.asname)

    @classmethod
    def of(
        cls,
        index: ModuleNodeIndex,
        *,
        source_module_name: str,
        owners: _AliasOwners,
    ) -> _SourceModuleAliasRetargets:
        candidates = tuple(
            alias
            for node in index.imports
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.asname is not None and alias.name == source_module_name
        )
        if not candidates:
            return cls(())
        return cls(
            tuple(
                (alias, owner)
                for alias in candidates
                if alias.asname is not None
                and index.bindings_in(alias.asname, scope := index.scope_of(alias)) == 1
                for owner in cls._sole_owner(index, alias.asname, scope, owners)
            )
        )

    @staticmethod
    def _sole_owner(
        index: ModuleNodeIndex,
        binding: str,
        scope: ast.AST | None,
        owners: _AliasOwners,
    ) -> tuple[str, ...]:
        """The one module every use of the alias's variable in `scope` reaches:
        the one owner all its names share, else the one module all of them can
        be read from once the move lands (a whole moved module, whose imports
        travel with its code), or none."""

        owned: list[frozenset[str]] = []
        readable: list[frozenset[str]] = []
        for node in index.uses_of(binding, scope):
            parent = index.parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                owned.append(owners.attribute_owners(parent.attr))
                readable.append(owners.attribute_readers(parent.attr))
            elif access := _NamedModuleMemberAccess.of(node, index.parents):
                owned.append(owners.access_owners(access))
                readable.append(owners.access_owners(access))
            else:
                return ()
        for reached in (owned, readable):
            common = frozenset.intersection(*reached) if reached else frozenset()
            if len(common) == 1:
                return tuple(common)
        return ()

    def localized_edits(
        self,
        index: ModuleNodeIndex,
        *,
        file_path: str,
        source: str,
        reader_module: str,
    ) -> tuple[NominalSourceEdit, ...]:
        """A module-level alias whose sole owner is the reading module itself
        reads names the reader now declares: each `alias.Name` becomes `Name`
        and the alias import goes, instead of the reader importing a module
        that is about to stop existing or importing itself."""

        local = tuple(
            alias
            for alias, owner in self.aliases
            if owner == reader_module and index.scope_of(alias) is None
        )
        if not local:
            return ()
        geometry = SourceTextGeometry(source)
        return (
            geometry.nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=parent.attr,
                    )
                    for alias in local
                    for node in index.uses_of(alias.asname or "", None)
                    if isinstance(parent := index.parents.get(node), ast.Attribute)
                    for start, end in (geometry.required_node_offsets(parent),)
                ),
            ),
            ModuleImportMutation.remove_bound_names(
                file_path=file_path,
                names=tuple(alias.asname for alias in local if alias.asname),
                rationale="The reader declares every name the alias read.",
            ),
        )

    def rewrite(self, file_path: str, source: str) -> NominalSourceEdit:
        geometry = SourceTextGeometry(source)
        return geometry.nominal_edit(
            file_path=file_path,
            replacements=tuple(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=end,
                    replacement_source=f"{owner} as {alias.asname}",
                )
                for alias, owner in self.aliases
                for start, end in (geometry.required_node_offsets(alias),)
            ),
        )


@dataclass(frozen=True)
class _SourceModuleObjectReads:
    """A consumer's `module.Name` reads of moved names through its bindings of the
    source module object, and its by-name accesses (`getattr(module, "Name")`,
    `monkeypatch.setattr(module, "Name", ...)`) of them. The move refuses when
    such a binding is used any other way, is rebound in the module's own scope,
    or when a name it would bind is already bound there to anything but that
    same moved declaration. A function, lambda or comprehension with its own
    variable of the binding's name shadows it only for its own uses
    (`ModuleNodeIndex.module_scope_names`). A module-level import that reaches
    the source module and is used only by the reads and accesses rewritten here
    has no reader left, and goes with them (`unread_bindings`)."""

    file_path: str
    reads: tuple[tuple[ast.Attribute, str], ...]
    accesses: tuple[tuple[ast.expr, _NamedModuleMemberAccess], ...]
    bound_names: frozenset[str]
    unread_bindings: tuple[tuple[str, ModuleImportScope], ...]
    imported_names: frozenset[str] = frozenset()

    @property
    def names(self) -> tuple[str, ...]:
        """The read names the consumer must now import: a name its top-level
        `from <source> import Name` already binds is rewritten with that import."""

        return tuple(
            sorted({name for _node, name in self.reads if name not in self.imported_names})
        )

    @classmethod
    def of(
        cls,
        parsed: ParsedModule,
        identity: PythonModulePathIdentity,
        *,
        source_module_name: str,
        moved_names: frozenset[str],
        file_path: str,
        retargeted: frozenset[str] = frozenset(),
    ) -> _SourceModuleObjectReads:
        source_parts = tuple(source_module_name.split("."))
        index = parsed.fact(ModuleNodeIndex)
        chains = frozenset(
            chain
            for chain in cls._bound_chains(index, identity, source_parts)
            if chain[0] not in retargeted
        )
        if not chains:
            return cls(file_path, (), (), frozenset(), ())
        bindings = index.bound_name_counts
        parents = index.parents
        reads: list[tuple[ast.Attribute, str]] = []
        accesses: list[tuple[ast.expr, _NamedModuleMemberAccess]] = []
        rewritten_loads: Counter[str] = Counter()
        for chain in chains:
            if index.module_scope_bindings(chain[0]) > 1:
                raise ValueError(
                    f"{file_path!r} rebinds {chain[0]!r}, which binds the source "
                    f"module {source_module_name!r}"
                )
            for node in index.module_scope_names(chain[0]):
                outermost, parts = cls._read_chain(node, parents, len(chain) + 1)
                if parts[: len(chain)] == chain and len(parts) > len(chain):
                    if parts[-1] in moved_names:
                        reads.append((outermost, parts[-1]))
                        rewritten_loads[chain[0]] += 1
                elif chain[: len(parts)] == parts and (
                    named := _NamedModuleMemberAccess.of(outermost, parents)
                ):
                    reached = (*parts, named.member_name)
                    if reached == chain:
                        raise ValueError(
                            f"{file_path!r} names {'.'.join(reached)!r}, which "
                            f"reaches the moved names, through {named.accessor!r}"
                        )
                    if parts == chain and named.member_name in moved_names:
                        accesses.append((outermost, named))
                        rewritten_loads[chain[0]] += 1
                elif parts == chain:
                    raise ValueError(
                        f"{file_path!r} uses the source module object "
                        f"{'.'.join(parts)!r} other than by reading an attribute"
                    )
        imported = frozenset(
            name
            for _node, name in reads
            if cls._imports_source_name(parsed, identity, source_module_name, name)
        )
        taken = tuple(
            sorted({name for _node, name in reads if bindings[name] and name not in imported})
        )
        if taken:
            raise ValueError(
                f"{file_path!r} already binds {taken!r}, which its source-module "
                "reads would be redirected to"
            )
        return cls(
            file_path,
            tuple(reads),
            tuple(accesses),
            frozenset(name for name, count in bindings.items() if count),
            cls._unread_bindings(parsed, chains, rewritten_loads),
            imported,
        )

    @staticmethod
    def _imports_source_name(
        parsed: ParsedModule,
        identity: PythonModulePathIdentity,
        source_module_name: str,
        name: str,
    ) -> bool:
        """Whether every binding of `name` is a top-level, runtime
        `from <source> import name`: the move rewrites that import to the name's
        owner, so `module.name` and `name` stay one object."""

        sites = parsed.fact(ModuleNodeIndex).binding_sites.get(name, ())
        return bool(sites) and all(
            isinstance(site, ast.ImportFrom)
            and site in parsed.module.body
            and identity.resolve_import_from_module(
                imported_module=site.module, level=site.level
            )
            == source_module_name
            and any(alias.name == name and alias.asname in (None, name) for alias in site.names)
            for site in sites
        )

    @staticmethod
    def _unread_bindings(
        parsed: ParsedModule,
        chains: frozenset[tuple[str, ...]],
        rewritten_loads: Counter[str],
    ) -> tuple[tuple[str, ModuleImportScope], ...]:
        """The module-level bindings that reach the source module and whose every
        load is a rewritten read or access, each with the import scope that binds
        it. A name several aliases bind (`import pkg.a, pkg.b`) stays: removing
        it would take the other modules' imports with it."""

        index = parsed.fact(ModuleNodeIndex)
        bound_by_alias = tuple(
            (name, scope)
            for scope in ModuleImportScope
            for name in scope.bindings(parsed.module)
        )
        alias_counts = Counter(name for name, _scope in bound_by_alias)
        scope_by_bound_name = {
            name: scope for name, scope in bound_by_alias if alias_counts[name] == 1
        }
        return tuple(
            sorted(
                (binding, scope_by_bound_name[binding])
                for binding in {chain[0] for chain in chains}
                if binding in scope_by_bound_name
                and rewritten_loads[binding]
                and rewritten_loads[binding] == index.module_scope_loads(binding)
            )
        )

    @staticmethod
    def _bound_chains(
        index: ModuleNodeIndex,
        identity: PythonModulePathIdentity,
        source_parts: tuple[str, ...],
    ) -> Iterator[tuple[str, ...]]:
        """The attribute chain that reaches the source module from each binding."""

        for node in index.imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    alias_parts = tuple(alias.name.split("."))
                    if alias.asname is None and alias_parts[0] == source_parts[0]:
                        yield source_parts
                    elif alias.asname is not None and source_parts[: len(alias_parts)] == alias_parts:
                        yield (alias.asname, *source_parts[len(alias_parts) :])
            elif isinstance(node, ast.ImportFrom):
                package = identity.resolve_import_from_module(
                    imported_module=node.module,
                    level=node.level,
                )
                for alias in node.names:
                    alias_parts = (*(package or "").split("."), alias.name)
                    if package and source_parts[: len(alias_parts)] == alias_parts:
                        yield (
                            ImportBoundNameProjection(node).alias_bound_name(alias),
                            *source_parts[len(alias_parts) :],
                        )

    @staticmethod
    def _read_chain(
        node: ast.Name,
        parents: dict[ast.AST, ast.AST],
        depth: int,
    ) -> tuple[ast.expr, tuple[str, ...]]:
        outermost: ast.expr = node
        parts = [node.id]
        while len(parts) < depth:
            parent = parents.get(outermost)
            if not (isinstance(parent, ast.Attribute) and parent.value is outermost):
                break
            outermost = parent
            parts.append(parent.attr)
        return outermost, tuple(parts)

    def rewrite(
        self,
        source: str,
        module_objects: tuple[tuple[ast.expr, str], ...],
    ) -> NominalSourceEdit:
        """Each read becomes its bare name; each by-name access's module object
        becomes the binding of the module it now targets."""

        geometry = SourceTextGeometry(source)
        return geometry.nominal_edit(
            file_path=self.file_path,
            replacements=tuple(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=end,
                    replacement_source=replacement,
                )
                for node, replacement in (*self.reads, *module_objects)
                for start, end in (geometry.required_node_offsets(node),)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveCarrier:
    """Shared source/destination carrier for closure-checked symbol moves."""

    source_path: str
    destination_path: str
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMoveRequest:
    """Agent-authored request for one dependency-checked symbol move."""

    selection: SourceTopLevelSymbolMoveSelection
    destination_path: str
    maximum_moved_symbol_count: int | None
    rationale: str = ""

    @property
    def source_path(self) -> str:
        return self.selection.source_path


@dataclass(frozen=True, kw_only=True)
class SourceTopLevelSymbolClosureMovePlan(SourceTopLevelSymbolClosureMoveCarrier):
    """Dependency-checked move plan for a set of top-level symbols."""

    source_blocks: tuple[MovedTopLevelDeclarationSource, ...]
    dependency_report: ModuleMoveDependencyReport
    source_binding_import_sources: tuple[str, ...]
    name_redirect: RelocationNameRedirect
    consumer_edits: tuple[NominalSourceEdit, ...]

    @classmethod
    def from_request(
        cls,
        request: SourceTopLevelSymbolClosureMoveRequest,
        context: CodemodSelectorContext,
        operation: ModuleSymbolMoveOperation,
    ) -> SourceTopLevelSymbolClosureMovePlan:
        source_table = ModuleSymbolTable(
            file_path=request.source_path,
            source=context.sources_by_file_path[request.source_path],
            module=context.module_nodes_by_file_path[request.source_path],
        )
        destination_table = ModuleSymbolTable(
            file_path=request.destination_path,
            source=context.sources_by_file_path[request.destination_path],
            module=context.module_nodes_by_file_path[request.destination_path],
        )
        declarations = request.selection.declarations
        moved_symbol_names = tuple(declaration.name for declaration in declarations)
        self_binding_removals = cls._destination_self_binding_removals(
            context,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        cls._validate_destination(
            destination_table,
            moved_symbol_names,
            excused_names=frozenset(
                name for removal in self_binding_removals for name in removal[1]
            ),
        )
        source_blocks = tuple(
            MovedTopLevelDeclarationSource.from_declaration(
                declaration,
                context.sources_by_file_path,
            )
            for declaration in declarations
        )
        report = cls._dependency_report(
            context,
            source_table,
            destination_table,
            request,
            operation,
            kept_reexports=cls._kept_reexports(context, source_table),
        )
        source_binding_import_sources = operation.source_binding_import_sources(
            context,
            source_table=source_table,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        import_graph = context.module_import_graph
        destination_module_name = import_graph.module_name_for_file_path(
            request.destination_path
        )
        source_module_name = import_graph.module_name_for_file_path(request.source_path)
        destination_localizations = (
            ()
            if destination_module_name is None or source_module_name is None
            else _SourceModuleAliasRetargets.of(
                destination_index := context.parsed_module_for_source_path(
                    request.destination_path
                ).fact(ModuleNodeIndex),
                source_module_name=source_module_name,
                owners=_DestinationOwners(destination_module_name, frozenset(moved_symbol_names)),
            ).localized_edits(
                destination_index,
                file_path=request.destination_path,
                source=context.sources_by_file_path[request.destination_path],
                reader_module=destination_module_name,
            )
        )
        name_redirect = RelocationNameRedirect(
            source_path=request.source_path,
            destinations={
                **cls._removed_binding_owners(context, report, request.destination_path),
                **{name: request.destination_path for name in moved_symbol_names},
            },
            reader_path=request.destination_path,
            reader_bindings=operation.reader_bindings(
                context, report, source_table, frozenset(moved_symbol_names)
            ),
        )
        consumer_edits = (
            *destination_localizations,
            *name_redirect.import_string_edits(context, request.destination_path),
            *(
                ModuleImportMutation.remove_names(
                    file_path=request.destination_path,
                    module_name=module_text,
                    names=names,
                    scope=scope,
                    rationale=request.rationale
                    or f"The destination now declares {names!r} itself.",
                )
                for module_text, names, scope in self_binding_removals
            ),
            *name_redirect.consumer_edits(
                context,
                excluded_paths=frozenset((request.source_path, request.destination_path)),
            ),
        )
        return cls(
            source_path=request.source_path,
            destination_path=request.destination_path,
            source_blocks=tuple(
                sorted(source_blocks, key=lambda block: block.source_start_line)
            ),
            dependency_report=report,
            source_binding_import_sources=source_binding_import_sources,
            name_redirect=name_redirect,
            consumer_edits=consumer_edits,
            rationale=request.rationale,
        )

    @classmethod
    def _removed_binding_owners(
        cls,
        context: CodemodSelectorContext,
        report: ModuleMoveDependencyReport,
        destination_path: str,
    ) -> dict[str, str]:
        """Source imports the move removes, each to where a reader that reached the
        name through the source reads it now."""

        return {
            dependency.name: cls._removed_binding_owner(context, dependency, destination_path)
            for dependency in report.source_removal_dependencies
        }

    @staticmethod
    def _removed_binding_owner(
        context: CodemodSelectorContext,
        dependency: ModuleMoveImportDependency,
        destination_path: str,
    ) -> str:
        """A name imported under its own name from an indexed module reads from the
        module declaring it, past modules that only re-import it. Any other removed
        binding (`import os`, `from os import sep`, an alias) names no indexed
        declaration; the move removes it only because the moved code alone read it,
        so the destination binds it now and the reader reads it there. Leaving
        those alone left `source.os` reading a binding the move took away."""

        identity = dependency.identity
        indexed = context.module_import_graph.unique_source_file_for_module_name(identity.module_name)
        if identity.imported_name != dependency.name or indexed is None:
            return destination_path
        owner = ImportedNameOwner.declaring_path(context, identity.module_name, dependency.name)
        if owner is None:
            raise ValueError(
                f"{dependency.name!r} is imported from indexed module {identity.module_name!r}, "
                "but no indexed module declares it: its re-imports cycle or leave the index"
            )
        return owner

    @staticmethod
    def _destination_self_binding_removals(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[tuple[str, tuple[str, ...], ModuleImportScope], ...]:
        """The destination's own imports of the moved declarations from the source:
        once the declarations land there, those imports would rebind them, so the
        move removes them instead of refusing the destination."""

        import_graph = context.module_import_graph
        source_module_name = import_graph.module_name_for_file_path(source_path)
        destination_file = import_graph.source_file_for_path(destination_path)
        if source_module_name is None or destination_file is None:
            return ()
        moved = frozenset(moved_symbol_names)
        removals: list[tuple[str, tuple[str, ...], ModuleImportScope]] = []
        for scope in ModuleImportScope:
            for statement in scope.import_statements(
                context.module_nodes_by_file_path[destination_path]
            ):
                if not isinstance(statement, ast.ImportFrom) or (
                    destination_file.module_path_identity.resolve_import_from_module(
                        imported_module=statement.module,
                        level=statement.level,
                    )
                    != source_module_name
                ):
                    continue
                names = tuple(
                    alias.name
                    for alias in statement.names
                    if alias.name in moved and alias.asname in (None, alias.name)
                )
                if names:
                    removals.append(
                        (ImportFromModuleName.from_node(statement).source, names, scope)
                    )
        return tuple(removals)

    @staticmethod
    def _validate_destination(
        destination_table: ModuleSymbolTable,
        moved_symbol_names: tuple[str, ...],
        *,
        excused_names: frozenset[str] = frozenset(),
    ) -> None:
        destination_names = destination_table.top_level_names | frozenset(
            destination_table.import_bindings_by_name
        )
        duplicate_names = tuple(
            name
            for name in moved_symbol_names
            if name in destination_names and name not in excused_names
        )
        if duplicate_names:
            raise ValueError(
                f"Destination {destination_table.file_path!r} already binds moved "
                "declarations "
                f"{duplicate_names!r}"
            )

    @staticmethod
    def _kept_reexports(
        context: CodemodSelectorContext,
        source_table: ModuleSymbolTable,
    ) -> frozenset[str]:
        """Source imports other files import through the source and that no
        redirect can point them past: the name is declared outside the index, or
        the source binds it under another name. Every other such import is removed
        with the move, and its importers read the name from its declaring module
        (`_removed_binding_owners`)."""

        source_path = source_table.file_path
        import_graph = context.module_import_graph
        source_module_name = import_graph.module_name_for_file_path(source_path)
        kept: set[str] = set()
        for source_file, parsed in FilesReachingModule(context, source_module_name):
            if source_file.file_path == source_path:
                continue
            for node in parsed.fact(ModuleNodeIndex).import_froms():
                if (
                    source_file.module_path_identity.resolve_import_from_module(
                        imported_module=node.module,
                        level=node.level,
                    )
                    != source_module_name
                ):
                    continue
                for alias in node.names:
                    binding = source_table.import_binding_for_dependency(
                        alias.name,
                        import_graph=import_graph,
                        permits_guarded_import=True,
                    )
                    if binding is not None and (
                        binding[1].imported_name != alias.name
                        or ImportedNameOwner.declaring_path(
                            context, binding[1].module_name, alias.name
                        )
                        is None
                    ):
                        kept.add(alias.name)
        return frozenset(kept)

    @classmethod
    def _dependency_report(
        cls,
        context: CodemodSelectorContext,
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        request: SourceTopLevelSymbolClosureMoveRequest,
        operation: ModuleSymbolMoveOperation,
        *,
        kept_reexports: frozenset[str] = frozenset(),
    ) -> ModuleMoveDependencyReport:
        selection = request.selection
        declarations = selection.declarations
        moved_names = frozenset(declaration.name for declaration in declarations)
        moved_dependencies = DeclarationDependencyProjection.from_declarations(
            tuple(declaration.node for declaration in declarations)
        )
        source_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            source_table.module
        )
        destination_annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            destination_table.module
        )
        import_graph = context.module_import_graph
        source_module_context_names = tuple(
            sorted(
                moved_dependencies.names
                & source_table.implicit_module_names
                - operation.module_context_names_following(context, source_table, moved_names)
            )
        )
        builtin_dependency_names = (
            moved_dependencies.names & source_table.unshadowed_builtin_names
        )
        destination_builtin_conflict_names = tuple(
            sorted(builtin_dependency_names & destination_table.explicit_names)
        )
        external_names = (
            moved_dependencies.names
            - source_table.unshadowed_builtin_names
            - source_table.implicit_module_names
        )
        permits_guarded_import_by_name = {
            name: (
                name not in moved_dependencies.execution_names
                and (
                    not source_annotation_mode.annotations_execute_at_declaration
                    or name not in moved_dependencies.evaluated_annotation_names
                )
            )
            for name in external_names
        }
        ambiguous_import_names = tuple(
            sorted(
                name
                for name in external_names
                if source_table.import_dependency_is_ambiguous(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
        )
        source_import_binding_by_name = {
            name: binding_and_identity
            for name in external_names
            if name not in ambiguous_import_names
            if (
                binding_and_identity := source_table.import_binding_for_dependency(
                    name,
                    import_graph=import_graph,
                    permits_guarded_import=permits_guarded_import_by_name[name],
                )
            )
            is not None
        }
        source_dependency_import_names = tuple(sorted(source_import_binding_by_name))
        resolved_import_names = frozenset(source_import_binding_by_name)
        ambiguous_import_name_set = frozenset(ambiguous_import_names)
        source_local_names = tuple(
            sorted(
                (
                    external_names
                    - moved_names
                    - resolved_import_names
                    - ambiguous_import_name_set
                )
                & source_table.top_level_names
            )
        )
        source_local_resolution = operation.source_local_dependency_resolution(
            import_graph,
            source_path=source_table.file_path,
            destination_table=destination_table,
            names=source_local_names,
        )
        unresolved_names = tuple(
            sorted(
                external_names
                - moved_names
                - resolved_import_names
                - ambiguous_import_name_set
                - source_table.top_level_names
            )
        )
        remaining_references = source_table.referenced_names_excluding(
            moved_names,
            source_dependency_import_names,
        )
        import_dependencies = tuple(
            ModuleMoveImportDependency(
                binding=binding,
                identity=identity,
                destination_import_required=(
                    not destination_table.satisfies_import_binding(
                        name,
                        identity,
                        binding.scope,
                        import_graph=import_graph,
                    )
                ),
                source_removal_required=(
                    binding.supports_bound_name_removal
                    and name not in remaining_references
                    and name not in source_table.explicit_reexport_bound_names
                    and name not in kept_reexports
                ),
                source_scopes=tuple(
                    dict.fromkeys(
                        (
                            binding.scope,
                            *(
                                bound.scope
                                for bound in source_table.import_bindings_by_name.get(
                                    name, ()
                                )
                            ),
                        )
                    )
                ),
            )
            for name, (binding, identity) in sorted(
                source_import_binding_by_name.items()
            )
        )
        destination_dependency_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if dependency.identity.is_destination_declaration(
                import_graph,
                destination_path=destination_table.file_path,
                bound_name=dependency.name,
            )
            and dependency.name in destination_table.top_level_names
        )
        destination_import_conflict_names = tuple(
            dependency.name
            for dependency in import_dependencies
            if destination_table.conflicts_with_import_binding(
                dependency.name,
                dependency.identity,
                import_graph=import_graph,
            )
        )
        return ModuleMoveDependencyReport(
            selection=selection,
            destination_path=destination_table.file_path,
            maximum_moved_symbol_count=request.maximum_moved_symbol_count,
            import_dependencies=import_dependencies,
            source_local_dependencies=source_local_resolution.import_dependencies,
            destination_dependency_names=destination_dependency_names,
            destination_insertion_line=destination_table.insertion_line_after_bindings(
                destination_dependency_names,
                (dependency.scope for dependency in import_dependencies),
            ),
            source_annotation_evaluation_mode=source_annotation_mode,
            destination_annotation_evaluation_mode=(destination_annotation_mode),
            moved_annotation_count=moved_dependencies.annotation_count,
            obstacles=(
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_LOCAL_DEPENDENCY,
                    source_local_resolution.unresolved_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.SOURCE_MODULE_CONTEXT_DEPENDENCY,
                    tuple(
                        operation.module_context_detail(context, name)
                        for name in source_module_context_names
                    ),
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_BUILTIN_CONFLICT,
                    destination_builtin_conflict_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.UNRESOLVED_DEPENDENCY,
                    unresolved_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.AMBIGUOUS_IMPORT_DEPENDENCY,
                    ambiguous_import_names,
                ),
                ModuleMoveObstacle(
                    ModuleMoveObstacleKind.DESTINATION_IMPORT_CONFLICT,
                    tuple(
                        sorted(
                            {
                                *destination_import_conflict_names,
                                *source_local_resolution.destination_conflict_names,
                            }
                        )
                    ),
                ),
                ModuleMoveObstacle.for_annotation_evaluation(
                    source_mode=source_annotation_mode,
                    destination_mode=destination_annotation_mode,
                    annotation_count=moved_dependencies.annotation_count,
                ),
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        self.dependency_report.require_clean()
        edits: list[NominalSourceEdit] = [
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=dependency.destination_source(
                        context.module_import_graph,
                        self.destination_path,
                    ),
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Ensure dependencies for moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for dependency in self.dependency_report.destination_import_dependencies
            ),
            *(
                ModuleImportMutation.from_source(
                    file_path=self.destination_path,
                    import_source=dependency.destination_import_source,
                    rationale=self.rationale
                    or (
                        "Ensure source-local dependencies for relocated symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
                for dependency in self.dependency_report.source_local_dependencies
            ),
            self.destination_insertion(context),
            *(
                block.deletion_replacement(
                    rationale=self.rationale,
                )
                for block in self.source_blocks
            ),
        ]
        edits.extend(
            (
                ModuleImportMutation.remove_bound_names(
                    file_path=self.source_path,
                    names=(dependency.name,),
                    scope=scope,
                    rationale=self.rationale
                    or (
                        "Remove imports used only by moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
            )
            for dependency in self.dependency_report.source_removal_dependencies
            for scope in dependency.source_scopes
        )
        edits.extend(
            ModuleImportMutation.from_source(
                file_path=self.source_path,
                import_source=import_source,
                rationale=self.rationale
                or (
                    "Preserve source bindings for moved symbols "
                    f"{self.dependency_report.moved_symbol_names!r}."
                ),
            )
            for import_source in self.source_binding_import_sources
        )
        edits.extend(self.consumer_edits)
        return tuple(edits)

    def destination_insertion(
        self,
        context: CodemodSelectorContext,
    ) -> SourceInsertion:
        destination_source = context.sources_by_file_path[self.destination_path]
        insertion_line = self.dependency_report.destination_insertion_line
        leading_boundary = self.source_blocks[0].declaration.destination_boundary
        return SourceInsertion(
            file_path=self.destination_path,
            insertion_line=insertion_line,
            inserted_lines=SourceTargetEditor.source_lines(
                self.destination_source(destination_source, insertion_line, context)
            ),
            leading_boundary=leading_boundary,
            rationale=self.rationale
            or (
                f"Move symbols {self.dependency_report.moved_symbol_names!r} "
                f"into {self.destination_path!r}."
            ),
        )

    def destination_source(
        self,
        destination_source: str,
        insertion_line: int,
        context: CodemodSelectorContext,
    ) -> str:
        rebase = _RelativeImportRebase.between(context, self.source_path, self.destination_path)
        moved_source = "\n\n\n".join(
            self.name_redirect.moved_text(
                context.module_import_graph,
                rebase.applied(block.moved_source.strip("\n")),
            )
            for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing.from_source(
            destination_source,
            insertion_line,
            inserted_source_is_import_block=False,
            boundary=self.source_blocks[0].declaration.destination_boundary,
        )
        return f"{spacing.leading_separator}{moved_source}{spacing.trailing_separator}"


@dataclass(frozen=True, kw_only=True)
class RetainedCallOperation(RepositorySourceReprovedOperation, ABC):
    """A module operation that rewrites strings naming the module, except in the
    claim of a call named in `retained_calls`: its first positional argument
    records the module's old identity on purpose (evidence that the path is gone,
    such as PathsAbsent's paths or PatternAbsent's pattern) inside an otherwise
    live file, and stays as written. The call's other arguments say where to
    look, and follow the move like any other mention: a search scope left on the
    old path would read a file that no longer exists and hold vacuously."""

    retained_calls: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RepositorySourceReprovedOperation, ABC):
    """Repository-proved destination contract for module-symbol moves."""

    destination_path: str = codemod_payload_field(SourcePathPayloadValueCodec())

    def module_context_names_following(
        self,
        context: CodemodSelectorContext,
        source_table: ModuleSymbolTable,
        moved_names: frozenset[str],
    ) -> frozenset[str]:
        """Module-context names whose meaning moves with the code. Into a module that
        already exists, none: `__name__` there names another module."""
        del context
        return frozenset()

    def moves_whole_module(self, context: CodemodSelectorContext) -> bool:
        """Whether the destination becomes the source module under a new name. A module
        that already exists stays another module, so a move into one never does."""
        del context
        return False

    def reader_bindings(
        self,
        context: CodemodSelectorContext,
        report: ModuleMoveDependencyReport,
        source_table: ModuleSymbolTable,
        moved_names: frozenset[str],
    ) -> frozenset[str]:
        """What the destination binds for a source alias's readers besides the moved
        declarations. Only a whole module's move makes the destination the module the
        alias meant, with its imports travelling and its following context names read
        there; any other move leaves those on the source that stays."""
        if not self.moves_whole_module(context):
            return frozenset()
        return frozenset(
            (
                *report.imported_dependency_names,
                *self.module_context_names_following(context, source_table, moved_names),
            )
        )

    def module_context_detail(self, context: CodemodSelectorContext, name: str) -> str:
        """How a module-context name the move leaves behind reads in its refusal."""
        del context
        return name

    def dependency_report(
        self,
        context: CodemodSelectorContext,
    ) -> ModuleMoveDependencyReport:
        return self.move_plan(context).dependency_report

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            dependency_report = self.required_reproof(
                lambda: self.dependency_report(context)
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        if dependency_report.is_clean:
            status = CodemodPreflightStatus.PASSED
            message = "Module symbol move dependency closure is clean"
        else:
            status = CodemodPreflightStatus.FAILED
            message = dependency_report.error_message
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=status,
                message=message,
                detail=dependency_report,
            ),
        )

    @abstractmethod
    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        """Resolve source-owned dependencies through this move's binding policy."""

        raise NotImplementedError

    @abstractmethod
    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return source bindings required after moving the declarations."""

        raise NotImplementedError

    def move_plan(
        self,
        context: CodemodSelectorContext,
    ) -> SourceTopLevelSymbolClosureMovePlan:
        """The move's plan for this source state, built once: preflight reads its
        dependency report and the edits read its source edits."""
        return context.retained_operation_derivation(
            (self, ModuleSymbolMoveOperation.move_plan),
            lambda: self._derived_move_plan(context),
        )

    def _derived_move_plan(
        self,
        context: CodemodSelectorContext,
    ) -> SourceTopLevelSymbolClosureMovePlan:
        source_path = self.required_source_path(context, self.operation_key())
        destination_path = SourcePathResolutionAuthority.from_source_index(
            self.destination_path,
            context.source_index,
        ).required_path()
        if source_path == destination_path:
            raise ValueError("Module symbol move destination must differ from source")
        return SourceTopLevelSymbolClosureMovePlan.from_request(
            SourceTopLevelSymbolClosureMoveRequest(
                selection=self.move_selection(context, source_path),
                destination_path=destination_path,
                maximum_moved_symbol_count=self.moved_symbol_count_limit(),
                rationale=self.rationale,
            ),
            context=context,
            operation=self,
        )

    def move_symbol_qualnames(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> tuple[str, ...]:
        """Return the declaration names derived from the current selection."""

        return self.move_selection(context, source_path).symbol_qualnames

    @abstractmethod
    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        """Return exact declarations to move from the current source state."""

        raise NotImplementedError

    @abstractmethod
    def moved_symbol_count_limit(self) -> int | None:
        """Return the explicit closure bound, or no bound for an exact selection."""

        raise NotImplementedError

    def move_source_edits(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_plan(context).source_edits(context)


@dataclass(frozen=True, kw_only=True)
class ExplicitModuleSymbolSelectionOperationABC(ModuleSymbolMoveOperation, ABC):
    """Operation whose payload explicitly selects every moved declaration."""

    symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def moved_symbol_count_limit(self) -> None:
        return None

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.exact(
            source_path,
            context.module_nodes_by_file_path[source_path],
            self.symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class DependencyClosureModuleSymbolSelectionOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Operation deriving a complete movable closure from semantic roots."""

    root_symbol_qualnames: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    maximum_moved_symbol_count: int = codemod_payload_field(
        RequiredIntegerPayloadValueCodec()
    )

    def moved_symbol_count_limit(self) -> int:
        return self.maximum_moved_symbol_count

    def move_selection(
        self,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> SourceTopLevelSymbolMoveSelection:
        return SourceTopLevelSymbolMoveSelection.dependency_closure(
            source_path,
            context.module_nodes_by_file_path[source_path],
            self.root_symbol_qualnames,
        )


@dataclass(frozen=True, kw_only=True)
class ExistingModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination already belongs to the source index."""

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.move_source_edits(context)


@dataclass(frozen=True)
class _ModuleSiblingData(ABC):
    """What a module reads about where it sits: files beside it through `__file__`, its
    top package's directory, and `__package__` as a value. A move that takes the whole
    module keeps a name following the code only while every read of it keeps its value
    there: a file beside the module is carried along, the top package's directory is
    the same from any module inside it, and `__package__` is unchanged only beside the
    source. Every other name stays behind, with why, and the move refuses."""

    stays: Mapping[str, str]
    """Each location name whose reads change at the destination, with why."""

    @property
    def following(self) -> frozenset[str]:
        """The module-context names whose meaning this data lets follow the code."""
        return frozenset(("__file__", "__package__")) - self.stays.keys()

    @abstractmethod
    def carry(
        self, context: CodemodSelectorContext, operation: RefactorRecipeOperation
    ) -> tuple[TextFileChange, ...]: ...

    def detail(self, name: str) -> str:
        """How a module-context name that stays behind reads in the refusal."""
        if name in self.stays:
            return f"{name} ({self.stays[name]})"
        return name

    @classmethod
    def of(
        cls, context: CodemodSelectorContext, source_path: str, destination_path: str
    ) -> _ModuleSiblingData:
        """Files are carried only when every `__file__` read is `Path(__file__).with_name(NAME)`,
        `Path(__file__).parent / NAME` for a literal NAME, or the top package's directory
        `Path(__file__).resolve().parents[(__package__ or "").count(".")]`; each NAME exists
        beside the source, no other indexed module beside it spells that NAME, and
        nothing of that name stands beside the destination yet."""

        source_dir = Path(source_path).parent
        destination_dir = Path(destination_path).parent
        parsed = context.parsed_module_for_source_path(source_path)
        module = parsed.module
        stays: dict[str, str] = {}
        try:
            reads = _SiblingFileReads.reads(parsed)
        except _UnrecognizedFileRead as read:
            reads = ()
            stays["__file__"] = f"it reads `__file__` as {read}, not as a file beside the module"
        located = frozenset(node for read in reads for node in read.package_reads)
        if located and not _TopPackageRead.keeps_value(
            context.module_import_graph, source_path, destination_path
        ):
            stays["__file__"] = stays["__package__"] = (
                "it reads its top package's directory, and the destination is in another top package"
            )
        if source_dir != destination_dir and any(
            isinstance(node, ast.Name) and node.id == "__package__" and node not in located
            for node in ast.walk(module)
        ):
            stays["__package__"] = "it reads `__package__`, which names another package at the destination"
        names = frozenset(name for read in reads for name in read.sibling_names)
        if not names:
            if not located:
                stays.setdefault("__file__", "it reads `__file__` without naming a file beside the module")
            return _NoSiblingCarry(stays)
        # A neighbour reaches the file only by spelling its name.
        for other_path, other_module in context.module_nodes_by_file_path.items():
            if other_path != source_path and Path(other_path).parent == source_dir:
                if shared := sorted(names & _SiblingFileReads.spelled(other_module)):
                    return _NoSiblingCarry({**stays, "__file__": f"{other_path!r} also names {shared!r}"})
        files: list[tuple[Path, Path]] = []
        for name in sorted(names):
            sibling = source_dir / name
            if not sibling.exists():
                return _NoSiblingCarry({**stays, "__file__": f"{sibling.as_posix()!r} does not exist"})
            if (destination_dir / name).exists():
                return _NoSiblingCarry(
                    {**stays, "__file__": f"{(destination_dir / name).as_posix()!r} already exists"}
                )
            members = (
                sorted(path for path in sibling.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
                if sibling.is_dir()
                else [sibling]
            )
            files.extend((member, destination_dir / member.relative_to(source_dir)) for member in members)
        try:
            texts = tuple(
                (source.as_posix(), destination.as_posix(), source.read_text(encoding="utf-8"))
                for source, destination in files
            )
        except UnicodeDecodeError as error:
            return _NoSiblingCarry({**stays, "__file__": f"a file it reads is not UTF-8 text ({error.reason})"})
        return _CarriedSiblingData(stays, texts)


@dataclass(frozen=True)
class _NoSiblingCarry(_ModuleSiblingData):
    """No file beside the module moves with it."""

    def carry(
        self, context: CodemodSelectorContext, operation: RefactorRecipeOperation
    ) -> tuple[TextFileChange, ...]:
        del context, operation
        return ()


@dataclass(frozen=True)
class _CarriedSiblingData(_ModuleSiblingData):
    files: tuple[tuple[str, str, str], ...]

    def carry(
        self, context: CodemodSelectorContext, operation: RefactorRecipeOperation
    ) -> tuple[TextFileChange, ...]:
        """Each file moves, and every data file the indexed code reads that spells
        a moved file's project path follows it."""

        return (
            *(
                change
                for source, destination, text in self.files
                for change in TextFileCarryEnd.carry(
                    operation, source_path=source, destination_path=destination, text=text
                )
            ),
            *_CodeReadDataFiles.path_edits(
                context,
                operation,
                tuple((source, destination) for source, destination, _text in self.files),
            ),
        )


class _LocationRead(ABC):
    """One read of where a module sits through `__file__`."""

    @property
    def sibling_names(self) -> frozenset[str]:
        """The files beside the module this read names."""
        return frozenset()

    @property
    def package_reads(self) -> frozenset[ast.Name]:
        """The `__package__` reads this read accounts for."""
        return frozenset()


@dataclass(frozen=True)
class _SiblingRead(_LocationRead):
    """`Path(__file__).with_name(NAME)` or `Path(__file__).parent / NAME`."""

    name: str

    @property
    def sibling_names(self) -> frozenset[str]:
        return frozenset((self.name,))


@dataclass(frozen=True)
class _TopPackageRead(_LocationRead):
    """`Path(__file__).resolve().parents[(__package__ or "").count(".")]`: the directory of
    the module's top package, the same from every module inside it."""

    package: ast.Name

    @property
    def package_reads(self) -> frozenset[ast.Name]:
        return frozenset((self.package,))

    @staticmethod
    def keeps_value(
        import_graph: SourceModuleImportGraph, source_path: str, destination_path: str
    ) -> bool:
        """Whether the read names the same directory from the destination, whose
        package is where it sits under the import root the source sits under."""
        name = import_graph.module_name_for_file_path(source_path)
        if name is None:
            raise ValueError(f"Source module identity is unavailable for {source_path!r}")
        source = Path(source_path)
        root = source.parents[name.count(".") + (source.name == "__init__.py")]
        destination = Path(destination_path)
        if not destination.is_relative_to(root):
            return False

        def directory(path: Path) -> Path:
            return path.parents[max(len(path.relative_to(root).parent.parts) - 1, 0)]

        return directory(source) == directory(destination)


class _UnrecognizedFileRead(ValueError):
    """A read of `__file__` other than naming a file beside the module."""


class _SiblingFileReads:
    """How a module reads where it sits through `__file__`."""

    @staticmethod
    def reads(parsed: ParsedModule) -> tuple[_LocationRead, ...]:
        parents = parsed.syntax_index.parent_by_node
        return tuple(
            _SiblingFileReads._read(node, parents)
            for node in ast.walk(parsed.module)
            if isinstance(node, ast.Name) and node.id == "__file__"
        )

    @staticmethod
    def spelled(module: ast.Module) -> frozenset[str]:
        """Every string constant the module spells, any one of which may name a file."""
        return frozenset(
            node.value
            for node in ast.walk(module)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )

    @staticmethod
    def _read(node: ast.Name, parents: Mapping[ast.AST, ast.AST]) -> _LocationRead:
        call = parents.get(node)
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "Path"
            and call.args == [node]
            and not call.keywords
        ):
            raise _UnrecognizedFileRead(ast.unparse(call))
        step = parents.get(call)
        if isinstance(step, ast.Attribute) and step.attr == "with_name":
            named = parents.get(step)
            if (
                isinstance(named, ast.Call)
                and named.func is step
                and len(named.args) == 1
                and isinstance(named.args[0], ast.Constant)
                and isinstance(named.args[0].value, str)
            ):
                return _SiblingRead(named.args[0].value)
        if isinstance(step, ast.Attribute) and step.attr == "parent":
            joined = parents.get(step)
            if (
                isinstance(joined, ast.BinOp)
                and isinstance(joined.op, ast.Div)
                and joined.left is step
                and isinstance(joined.right, ast.Constant)
                and isinstance(joined.right.value, str)
            ):
                return _SiblingRead(joined.right.value)
        if isinstance(step, ast.Attribute) and step.attr == "resolve":
            resolved = parents.get(step)
            ancestors = parents.get(resolved)
            match resolved, ancestors, parents.get(ancestors):
                case (
                    ast.Call(args=[], keywords=[]) as resolution,
                    ast.Attribute(attr="parents") as directories,
                    ast.Subscript(
                        slice=ast.Call(
                            func=ast.Attribute(
                                value=ast.BoolOp(
                                    op=ast.Or(),
                                    values=[ast.Name(id="__package__") as package, ast.Constant(value="")],
                                ),
                                attr="count",
                            ),
                            args=[ast.Constant(value=".")],
                            keywords=[],
                        )
                    ) as indexed,
                ) if resolution.func is step and indexed.value is directories:
                    return _TopPackageRead(package)
        raise _UnrecognizedFileRead(ast.unparse(step or call))


@dataclass(frozen=True, kw_only=True)
class NewModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination source is created atomically."""

    def module_context_names_following(
        self,
        context: CodemodSelectorContext,
        source_table: ModuleSymbolTable,
        moved_names: frozenset[str],
    ) -> frozenset[str]:
        """When every name the source declares moves into the module this move creates,
        the moved code is that module under its new name, so `__name__` (a logger's, say)
        names the destination and `__doc__` (an argparse description) reads the
        docstring the destination opens with. `__file__` and `__package__` follow only
        while every read of them keeps its value at the destination (_ModuleSiblingData)."""
        if not source_table.declared_names <= moved_names:
            return frozenset()
        following = frozenset(("__name__",)) | self.sibling_data(context).following
        if self.carries_source_docstring:
            return following | {"__doc__"}
        return following

    @property
    def carries_source_docstring(self) -> bool:
        """The created module opens with the source's docstring unless the caller
        writes its source or docstring."""
        return self.destination_source is None and self.destination_docstring is None

    def moves_whole_module(self, context: CodemodSelectorContext) -> bool:
        source_path = self.required_source_path(context, self.operation_key())
        source_table = ModuleSymbolTable(
            file_path=source_path,
            source=context.sources_by_file_path[source_path],
            module=context.module_nodes_by_file_path[source_path],
        )
        moved_names = frozenset(
            declaration.name
            for declaration in self.move_selection(context, source_path).declarations
        )
        return source_table.declared_names <= moved_names

    def module_context_detail(self, context: CodemodSelectorContext, name: str) -> str:
        if not self.moves_whole_module(context):
            return name
        return self.sibling_data(context).detail(name)

    def sibling_data(self, context: CodemodSelectorContext) -> _ModuleSiblingData:
        source_path = self.required_source_path(context, self.operation_key())
        return _ModuleSiblingData.of(context, source_path, self.destination_path)

    def text_file_changes(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[TextFileChange, ...]:
        """The data beside the source that moves with it, when the whole module moves."""

        if not self.moves_whole_module(context):
            return super().text_file_changes(context)
        return (*super().text_file_changes(context), *self.sibling_data(context).carry(context, self))

    destination_source: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )
    destination_docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return self.required_reproof(
            lambda: (
                SourceFileCreation.from_operation(
                    self,
                    requested_path=self.destination_path,
                    source_index=context.source_index,
                    source=self.initial_destination_source(context),
                ),
                *(
                    SourceFileCreation.from_operation(
                        self,
                        requested_path=init_path,
                        source_index=context.source_index,
                        source="",
                    )
                    for init_path in self.undeclared_package_inits(context)
                ),
            )
        )

    def undeclared_package_inits(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        """The `__init__.py` of each directory the destination creates inside a
        declared package: a module written under `pkg/` (which has its
        `__init__.py`) into `pkg/sub/` makes `sub` a package the way its parent
        is. Under no declared package (a top-level or namespace layout), none."""

        sources = context.sources_by_file_path
        missing: list[str] = []
        directory = PurePosixPath(self.destination_path).parent
        while directory != directory.parent:
            init_path = (directory / PACKAGE_INIT_FILE_NAME).as_posix()
            if init_path in sources:
                return tuple(reversed(missing))
            if init_path != self.destination_path:
                missing.append(init_path)
            directory = directory.parent
        return ()

    def initial_destination_source(self, context: CodemodSelectorContext) -> str:
        """Resolve caller source, or the caller's module docstring followed by the
        source module's derived annotation policy."""

        if self.destination_source is not None:
            return self.destination_source
        source_path = self.required_source_path(context, self.operation_key())
        source_module = context.module_nodes_by_file_path[source_path]
        prelude = ModuleAnnotationEvaluationMode.from_module(source_module).new_module_prelude()
        if self.destination_docstring is not None:
            return f'"""{self.destination_docstring}"""\n\n{prelude}'
        docstring = source_module.body[0] if source_module.body else None
        if (
            self.carries_source_docstring
            and isinstance(docstring, ast.Expr)
            and isinstance(docstring.value, ast.Constant)
            and isinstance(docstring.value.value, str)
            and self.moves_whole_module(context)
        ):
            return f"{ast.get_source_segment(context.sources_by_file_path[source_path], docstring)}\n\n{prelude}"
        return prelude

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return (
            *self.source_file_creations(context),
            *self.move_source_edits(context),
        )


@dataclass(frozen=True, kw_only=True)
class SourceBindingPreservingModuleSymbolMoveOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Move declarations while preserving public or locally required bindings."""

    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        del import_graph, source_path, destination_table
        return ModuleMoveSourceLocalDependencyResolution.blocked(names)

    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        import_graph = context.module_import_graph
        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        retained_reference_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        )
        return tuple(
            (
                import_graph.required_reexport_source
                if export_contract.exposure_for(symbol_name).blocks_closed_boundary
                else import_graph.required_import_source
            )(
                importing_file_path=source_path,
                imported_file_path=destination_path,
                imported_name=symbol_name,
            )
            for symbol_name in moved_symbol_names
            if (
                export_contract.exposure_for(symbol_name).blocks_closed_boundary
                or symbol_name in retained_reference_names
            )
        )


@dataclass(frozen=True, kw_only=True)
class SourceBindingRelocatingModuleSymbolMoveOperationABC(
    ModuleSymbolMoveOperation,
    ABC,
):
    """Move declarations only when their former bindings can be removed."""

    def text_file_changes(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[TextFileChange, ...]:
        """An entry point or setting naming a moved name on the source, which no
        longer binds it, names the name's owner instead."""

        source_path = self.required_source_path(context, self.operation_key())
        redirect = self.move_plan(context).name_redirect
        return (
            *super().text_file_changes(context),
            *_RelocatedNameStrings(
                source_path,
                redirect.source_module_name(context.module_import_graph),
                redirect.destinations,
                None,
            ).manifest_edits(context, self),
        )

    def source_local_dependency_resolution(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        source_path: str,
        destination_table: ModuleSymbolTable,
        names: tuple[str, ...],
    ) -> ModuleMoveSourceLocalDependencyResolution:
        source_module_name = import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(
                f"Source module identity is unavailable for {source_path!r}"
            )
        identities = {
            name: FromModuleImportBindingIdentity(
                module_name=source_module_name,
                member_name=name,
            )
            for name in names
        }
        conflict_names = tuple(
            name
            for name, identity in identities.items()
            if destination_table.conflicts_with_import_binding(
                name,
                identity,
                import_graph=import_graph,
            )
        )
        conflict_name_set = frozenset(conflict_names)
        import_dependencies = tuple(
            ModuleMoveSourceLocalDependency.from_source_declaration(
                import_graph,
                source_path=source_path,
                destination_path=destination_table.file_path,
                name=name,
            )
            for name, identity in identities.items()
            if name not in conflict_name_set
            if not destination_table.satisfies_import_binding(
                name,
                identity,
                ModuleImportScope.RUNTIME,
                import_graph=import_graph,
            )
        )
        return ModuleMoveSourceLocalDependencyResolution(
            import_dependencies=import_dependencies,
            destination_conflict_names=conflict_names,
        )

    def source_binding_import_sources(
        self,
        context: CodemodSelectorContext,
        *,
        source_table: ModuleSymbolTable,
        source_path: str,
        destination_path: str,
        moved_symbol_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """The source imports back, without re-exporting, each moved name it still
        reads: every other importer is redirected to the destination. Refused when
        the moved declarations also read names the source keeps defining, since
        the two modules would then import each other."""

        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        blocked_names = export_contract.blocked_relocations(moved_symbol_names)
        if blocked_names:
            raise ValueError(
                "Source export contract does not prove binding relocation for "
                f"{blocked_names!r}"
            )
        retained_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        ).intersection(moved_symbol_names)
        if not retained_names:
            return ()
        defined_in_source = (
            source_table.top_level_names
            - frozenset(source_table.import_bindings_by_name)
            - frozenset(moved_symbol_names)
        )
        read_back = tuple(
            sorted(
                {
                    node.id
                    for name in moved_symbol_names
                    for statement in source_table.binding_statements(name)
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name)
                }
                & defined_in_source
            )
        )
        if read_back:
            raise ValueError(
                "Relocated symbols remain referenced by their source module "
                f"{tuple(sorted(retained_names))!r} while reading names it keeps "
                f"{read_back!r}: source and destination would import each other"
            )
        return tuple(
            context.module_import_graph.required_import_source(
                importing_file_path=source_path,
                imported_file_path=destination_path,
                imported_name=name,
            )
            for name in sorted(retained_names)
        )


@dataclass(frozen=True, kw_only=True)
class MoveSymbolsToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Move an explicitly complete symbol set into an existing module."""


@dataclass(frozen=True, kw_only=True)
class RelocateSymbolsToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC,
):
    """Move exact symbols and remove their former source-module bindings."""


@dataclass(frozen=True, kw_only=True)
class RelocateSymbolsToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingRelocatingModuleSymbolMoveOperationABC,
):
    """Create a module, move exact symbols, and remove their former bindings."""


@dataclass(frozen=True, kw_only=True)
class MoveSymbolClosureToModuleOperation(
    ExistingModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Move a root-derived dependency closure into an existing module."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolsToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    ExplicitModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Create a module and move an explicitly complete symbol set into it."""


@dataclass(frozen=True, kw_only=True)
class ExtractSymbolClosureToNewModuleOperation(
    NewModuleSymbolMoveOperationABC,
    DependencyClosureModuleSymbolSelectionOperationABC,
    SourceBindingPreservingModuleSymbolMoveOperationABC,
):
    """Create a module and derive the moved closure from semantic roots."""


@dataclass(frozen=True)
class ImportedNameOwner:
    """The indexed module that declares an imported name, followed through the
    modules that only re-import it; None when the name comes from outside the index."""

    @staticmethod
    def declaring_path(
        context: CodemodSelectorContext,
        module_name: str | None,
        name: str,
    ) -> str | None:
        """The indexed module declaring `name`, through modules that only rebind it."""

        import_graph = context.module_import_graph
        seen: set[str] = set()
        while module_name is not None and module_name not in seen:
            seen.add(module_name)
            source_file = import_graph.unique_source_file_for_module_name(module_name)
            if source_file is None:
                return None
            table = ModuleSymbolTable(
                file_path=source_file.file_path,
                source=context.sources_by_file_path[source_file.file_path],
                module=context.module_nodes_by_file_path[source_file.file_path],
            )
            if name in table.top_level_names:
                return source_file.file_path
            rebinding = next(
                (
                    node
                    for scope in ModuleImportScope
                    for node in scope.import_statements(table.module)
                    if isinstance(node, ast.ImportFrom)
                    and any(alias.name == name and alias.asname in (None, name) for alias in node.names)
                ),
                None,
            )
            if rebinding is None:
                return source_file.file_path
            module_name = source_file.module_path_identity.resolve_import_from_module(
                imported_module=rebinding.module,
                level=rebinding.level,
            )
        return None


@dataclass(frozen=True)
class _DissolvedModuleBindings:
    """What a module that declares nothing still binds: each imported name and
    the indexed module that declares it, following re-export chains."""

    source_path: str
    owners: Mapping[str, str]
    foreign_names: frozenset[str]

    @classmethod
    def of(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
    ) -> _DissolvedModuleBindings:
        module = context.module_nodes_by_file_path[source_path]
        for index, statement in enumerate(module.body):
            if not cls._binds_only_imports(statement, first=index == 0):
                raise ValueError(
                    f"{source_path!r} still declares {ast.unparse(statement).splitlines()[0]!r} "
                    f"at line {statement.lineno}; relocate its declarations first"
                )
        owners: dict[str, str] = {}
        foreign: set[str] = set()
        for name, binding_module in cls._imported_names(context, source_path, module):
            owner = ImportedNameOwner.declaring_path(context, binding_module, name)
            if owner is None:
                foreign.add(name)
            else:
                owners[name] = owner
        return cls(source_path, owners, frozenset(foreign))

    def run_modules(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        """The modules importing the dissolved module runs, in its order: each
        module its top-level imports name, and each submodule a `from` import
        names. A condition other than `TYPE_CHECKING` or a script guard would run
        some of them only sometimes, so it is refused."""

        module = context.module_nodes_by_file_path[self.source_path]
        guards = TypeCheckingGuardProjection.from_module(module).guards
        for statement in module.body:
            if isinstance(statement, ast.If) and statement not in guards and not _ScriptEntry.is_guard(statement):
                raise ValueError(
                    f"{self.source_path!r} imports under the condition "
                    f"{ast.unparse(statement.test)!r} at line {statement.lineno}"
                )
        import_graph = context.module_import_graph
        source_file = import_graph.source_file_for_path(self.source_path)
        if source_file is None:
            raise ValueError(f"Source module identity is unavailable for {self.source_path!r}")
        known = import_graph.known_module_names
        run: list[str] = []
        for statement in ModuleImportScope.RUNTIME.import_statements(module):
            if isinstance(statement, ast.Import):
                run.extend(alias.name for alias in statement.names)
                continue
            if statement.module == "__future__":
                continue
            imported_module = source_file.module_path_identity.resolve_import_from_module(
                imported_module=statement.module, level=statement.level
            )
            if imported_module is None:
                raise ValueError(
                    f"{self.source_path!r} imports {ast.unparse(statement)!r}, "
                    "whose module does not resolve"
                )
            run.append(imported_module)
            run.extend(
                f"{imported_module}.{alias.name}"
                for alias in statement.names
                if f"{imported_module}.{alias.name}" in known
            )
        return tuple(dict.fromkeys(run))

    @staticmethod
    def _binds_only_imports(statement: ast.stmt, *, first: bool) -> bool:
        if isinstance(statement, (ast.Import, ast.ImportFrom)) or _ScriptEntry.is_guard(statement):
            return True
        if first and isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and isinstance(statement.value.value, str):
            return True
        if isinstance(statement, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            return True
        return (
            isinstance(statement, ast.If)
            and not statement.orelse
            and all(isinstance(inner, (ast.Import, ast.ImportFrom)) for inner in statement.body)
        )

    @staticmethod
    def _imported_names(
        context: CodemodSelectorContext,
        source_path: str,
        module: ast.Module,
    ) -> Iterator[tuple[str, str | None]]:
        identity = context.module_import_graph.source_file_for_path(
            source_path
        ).module_path_identity
        for scope in ModuleImportScope:
            for statement in scope.import_statements(module):
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        yield ImportBoundNameProjection(statement).alias_bound_name(alias), None
                    continue
                if statement.module == "__future__":
                    continue
                imported_module = identity.resolve_import_from_module(
                    imported_module=statement.module,
                    level=statement.level,
                )
                for alias in statement.names:
                    if alias.name == "*":
                        raise ValueError(
                            f"{source_path!r} star-imports {imported_module!r}; "
                            "its rebound names cannot be listed"
                        )
                    if alias.asname not in (None, alias.name):
                        raise ValueError(
                            f"{source_path!r} rebinds {alias.name!r} as {alias.asname!r}; "
                            "its consumers cannot import that name from an owner"
                        )
                    yield alias.name, imported_module


@dataclass(frozen=True)
class _ScriptEntry:
    """A module's `if __name__ == "__main__":` block: what running the module as
    a script does. Dissolving the module moves it to the end of the successor,
    which every `-m` string naming the module now names; each name it reads must
    be one the successor declares."""

    statement: ast.If

    @staticmethod
    def is_guard(statement: ast.stmt) -> bool:
        test = statement.test if isinstance(statement, ast.If) else None
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )

    @classmethod
    def of(cls, module: ast.Module) -> tuple[_ScriptEntry, ...]:
        return tuple(
            cls(cast(ast.If, statement)) for statement in module.body if cls.is_guard(statement)
        )

    def successor_edits(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        successor_path: str | None,
    ) -> tuple[NominalSourceEdit, ...]:
        """The block at the end of the successor, and the imports it needs there.
        Each name it runs is the source's import of it, the same proved
        dependency a relocated declaration carries: one the successor declares or
        already imports needs nothing, one it binds to another object refuses,
        and every other one is imported into the successor from where it is
        declared. A name the source does not bind refuses."""

        where = f"{source_path!r}:{self.statement.lineno}"
        if successor_path is None:
            raise ValueError(f"{where} runs the module as a script; declare its successor_path")
        source_module = context.module_nodes_by_file_path[source_path]
        successor_module = context.module_nodes_by_file_path[successor_path]
        if _ScriptEntry.of(successor_module):
            raise ValueError(f"{successor_path!r} already runs as a script; {where} cannot join it")
        import_graph = context.module_import_graph
        source_table = ModuleSymbolTable(
            file_path=source_path,
            source=context.sources_by_file_path[source_path],
            module=source_module,
        )
        successor_table = ModuleSymbolTable(
            file_path=successor_path,
            source=context.sources_by_file_path[successor_path],
            module=successor_module,
        )
        imports: list[ModuleImportMutation] = []
        unbound: list[str] = []
        for name in sorted(self.runs(source_module)):
            resolved = source_table.import_binding_for_dependency(
                name, import_graph=import_graph, permits_guarded_import=False
            )
            if resolved is None:
                unbound.append(name)
                continue
            binding, identity = resolved
            if identity.is_destination_declaration(
                import_graph, destination_path=successor_path, bound_name=name
            ) or successor_table.satisfies_import_binding(
                name, identity, binding.scope, import_graph=import_graph
            ):
                continue
            if successor_table.conflicts_with_import_binding(name, identity, import_graph=import_graph):
                raise ValueError(
                    f"{where} runs {name!r}, which {successor_path!r} binds to another object"
                )
            imports.append(
                ModuleImportMutation.from_source(
                    file_path=successor_path,
                    import_source=identity.source_for(
                        import_graph,
                        importing_file_path=successor_path,
                        scope=binding.scope,
                        bound_name=name,
                    ),
                    rationale=f"The script entry moved from {source_path!r} runs {name!r}.",
                )
            )
        if unbound:
            raise ValueError(
                f"{where} runs {unbound!r} as a script, which the source "
                f"{source_path!r} does not bind"
            )
        successor_source = context.sources_by_file_path[successor_path]
        return (
            *imports,
            SourceInsertion(
                file_path=successor_path,
                insertion_line=len(successor_source.splitlines()) + 1,
                inserted_lines=SourceTargetEditor.source_lines(
                    "\n\n"
                    + ast.get_source_segment(context.sources_by_file_path[source_path], self.statement)
                    + "\n"
                ),
                rationale=f"Running {source_path!r} as a script now runs {successor_path!r}.",
            ),
        )

    def runs(self, module: ast.Module) -> frozenset[str]:
        """The names the block reads from the module: its loads, less the names it
        binds itself and the builtins the module does not shadow."""

        loads = {
            node.id
            for node in ast.walk(self.statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.statement.body)
        module_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        builtin = frozenset(dir(builtins)) - module_bound
        return frozenset(loads - bound - builtin - {"__name__"})


class _GluedModuleName:
    """`f"...pkg.{name}"`: a dotted constant prefix with one name formatted onto it."""

    _PREFIX = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\.)$")

    @classmethod
    def sites(cls, index: ModuleNodeIndex) -> Iterator[tuple[ast.JoinedStr, str, str]]:
        for node in index.parents:
            match node:
                case ast.JoinedStr(values=values):
                    pass
                case _:
                    continue
            for before, part, after in zip(values, values[1:], (*values[2:], None)):
                match before, part, after:
                    case (
                        ast.Constant(value=str(text)),
                        ast.FormattedValue(value=ast.Name(id=name), format_spec=None),
                        None | ast.Constant(value=str()),
                    ) if (found := cls._PREFIX.search(text)) and not (
                        after is not None and re.match(r"[\w.]", after.value)
                    ):
                        yield node, found.group(1), name

    @classmethod
    def bound_literals(
        cls, index: ModuleNodeIndex, node: ast.AST, name: str
    ) -> tuple[ast.Constant, ...] | None:
        """The string literals `name` takes where `node` reads it, or None when
        the nearest binding is not a parametrize value or a literal iterable."""

        while (node := index.parents.get(node)) is not None:
            binding = _NameBinding.decoded(node)
            if binding.binds(name):
                return binding.literals(name)
        return None


class _NameBinding(ForeignAdapter):
    """An enclosing node, asked whether it binds a name and to which literals."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> _NameBinding:
        """Total: `_NoBinding` adapts `object`."""

        return cast(_NameBinding, super().decoded(node))

    def binds(self, name: str) -> bool:
        return False

    def literals(self, name: str) -> tuple[ast.Constant, ...] | None:
        return None


class _NoBinding(_NameBinding):
    adapts = object


class _ForBinding(_NameBinding):
    """`for name in (...)`."""

    adapts = ast.For

    def binds(self, name: str) -> bool:
        return ExpressionReference.decoded(self.node.target).identifier == name

    def literals(self, name: str) -> tuple[ast.Constant, ...] | None:
        return _LiteralSequence.decoded(self.node.iter).literals


class _ComprehensionBinding(_NameBinding):
    """`[... for name in (...)]`: the generator that binds the name."""

    adapts = ast.ListComp

    def _generator(self, name: str) -> ast.comprehension | None:
        return next(
            (
                generator
                for generator in self.node.generators
                if ExpressionReference.decoded(generator.target).identifier == name
            ),
            None,
        )

    def binds(self, name: str) -> bool:
        return self._generator(name) is not None

    def literals(self, name: str) -> tuple[ast.Constant, ...] | None:
        generator = self._generator(name)
        return None if generator is None else _LiteralSequence.decoded(generator.iter).literals


class _SetComprehensionBinding(_ComprehensionBinding):
    adapts = ast.SetComp


class _GeneratorBinding(_ComprehensionBinding):
    adapts = ast.GeneratorExp


class _DictComprehensionBinding(_ComprehensionBinding):
    adapts = ast.DictComp


class _LambdaBinding(_NameBinding):
    """A lambda parameter: bound to whatever a caller passes, never traced."""

    adapts = ast.Lambda

    def binds(self, name: str) -> bool:
        return name in {argument.arg for argument in self.node.args.args}


class _FunctionBinding(_LambdaBinding):
    """A function parameter: traced through a `pytest.mark.parametrize` decorator."""

    adapts = ast.FunctionDef

    def literals(self, name: str) -> tuple[ast.Constant, ...] | None:
        return next(
            (
                literals
                for decorator in self.node.decorator_list
                for literals in (_parametrized(decorator, name),)
                if literals is not None
            ),
            None,
        )


class _AsyncFunctionBinding(_FunctionBinding):
    adapts = ast.AsyncFunctionDef


def _parametrized(decorator: ast.expr, name: str) -> tuple[ast.Constant, ...] | None:
    match decorator:
        case ast.Call(
            func=ast.Attribute(attr="parametrize"),
            args=[ast.Constant(value=str(argnames)), rows_node, *_],
        ):
            names = [part.strip() for part in argnames.split(",")]
        case _:
            return None
    rows = _LiteralSequence.decoded(rows_node).elements
    if name not in names or rows is None:
        return None
    column = names.index(name)
    cells = tuple(_ParametrizeRow.decoded(row).cells(len(names)) for row in rows)
    if any(len(row_cells) <= column for row_cells in cells):
        return None
    return _LiteralSequence(ast.Tuple(elts=[row_cells[column] for row_cells in cells])).literals


class _ParametrizeRow(ForeignAdapter):
    """One row of a parametrize table, read as its cells."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> _ParametrizeRow:
        """Total: `_ScalarRow` adapts `object`."""

        return cast(_ParametrizeRow, super().decoded(node))

    def cells(self, width: int) -> tuple[ast.expr, ...]:
        return (self.node,)


class _ScalarRow(_ParametrizeRow):
    adapts = object


class _ParamRow(_ParametrizeRow):
    """`pytest.param(a, b, id=...)`."""

    adapts = ast.Call

    def cells(self, width: int) -> tuple[ast.expr, ...]:
        return tuple(self.node.args)


class _TupleRow(_ParametrizeRow):
    """`(a, b)` for two argnames; a single argname takes the tuple itself."""

    adapts = ast.Tuple

    def cells(self, width: int) -> tuple[ast.expr, ...]:
        return tuple(self.node.elts) if width > 1 else (self.node,)


class _ListRow(_TupleRow):
    adapts = ast.List


class _LiteralSequence(ForeignAdapter):
    """An expression read as a literal sequence of strings."""

    __registry_key__ = "adapts"

    @classmethod
    def decoded(cls, node: object) -> _LiteralSequence:
        """Total: `_OpaqueSequence` adapts `object`."""

        return cast(_LiteralSequence, super().decoded(node))

    @property
    def elements(self) -> tuple[ast.expr, ...] | None:
        return tuple(self.node.elts)

    @property
    def literals(self) -> tuple[ast.Constant, ...] | None:
        """Every element when all are string constants, else None."""

        elements = self.elements
        if elements is None:
            return None
        literals: list[ast.Constant] = []
        for node in elements:
            match node:
                case ast.Constant(value=str()):
                    literals.append(node)
                case _:
                    return None
        return tuple(literals)


class _TupleSequence(_LiteralSequence):
    adapts = ast.Tuple


class _ListSequence(_LiteralSequence):
    adapts = ast.List


class _SetSequence(_LiteralSequence):
    adapts = ast.Set


class _OpaqueSequence(_LiteralSequence):
    adapts = object

    @property
    def elements(self) -> tuple[ast.expr, ...] | None:
        return None


class _ProjectManifests:
    """The `pyproject.toml` governing each indexed source: the nearest one in
    the source's directory or above it."""

    file_name: ClassVar[str] = "pyproject.toml"

    @classmethod
    def governing(cls, file_paths: Iterable[str]) -> tuple[Path, ...]:
        manifests: set[Path] = set()
        for directory in {Path(file_path).parent for file_path in file_paths}:
            for candidate in (directory, *directory.parents):
                manifest = candidate / cls.file_name
                if manifest.is_file():
                    manifests.add(manifest)
                    break
        return tuple(sorted(manifests))


@dataclass(frozen=True)
class DataFilePathLiterals(ParsedModuleFact):
    """The string constants of one module that spell a data file's path, such as
    `"stack/native/repository-inputs.json"`. A fact of the parse, read from its
    `ModuleNodeIndex`, so an unchanged module is read once per plan."""

    literal: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:[\w.-]+/)+[\w.-]+\.(?:json|toml|ya?ml|cfg|ini|txt)"
    )
    paths: frozenset[str]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> DataFilePathLiterals:
        return cls(
            frozenset(
                node.value
                for node in parsed_module.fact(ModuleNodeIndex).string_constants
                if isinstance(node.value, str) and cls.literal.fullmatch(node.value)
            )
        )


class _CodeReadDataFiles:
    """Data files the indexed code reads by a literal path under its project root,
    such as `root / "stack/native/repository-inputs.json"`. The program consumes
    them, so a project path spelled inside one is a declaration that must stay
    true; prose and records no code reads keep what they said. Each file is read
    as the plan's state has it, so a later stage's carry edits an earlier stage's
    rewrite, not the file on disk."""

    @classmethod
    def path_edits(
        cls,
        context: CodemodSelectorContext,
        operation: RefactorRecipeOperation,
        moves: tuple[tuple[str, str], ...],
    ) -> tuple[TextFileEdit, ...]:
        snapshot = context.execution_snapshot()
        literals = frozenset().union(
            *(module.fact(DataFilePathLiterals).paths for module in snapshot.parsed_modules)
        )
        edits: list[TextFileEdit] = []
        for manifest in _ProjectManifests.governing(snapshot.sources_by_file_path):
            root = manifest.parent
            renames = tuple(
                (Path(source).relative_to(root).as_posix(), Path(destination).relative_to(root).as_posix())
                for source, destination in moves
                if Path(source).is_relative_to(root) and Path(destination).is_relative_to(root)
            )
            for data in sorted({root / literal for literal in literals}):
                for file_path, text in snapshot.text_file(data).sources(data.as_posix()).items():
                    replacements = tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=match.start(), end_offset=match.end(), replacement_source=new
                        )
                        for old, new in renames
                        for match in re.finditer(rf"(?<![\w./-]){re.escape(old)}(?![\w./-])", text)
                    )
                    if replacements:
                        edits.append(
                            TextFileEdit(
                                operation_type=type(operation),
                                file_path=file_path,
                                source=text,
                                replacements=replacements,
                                rationale=operation.rationale_text(
                                    f"{file_path!r} names a carried file by its project path."
                                ),
                            )
                        )
        return tuple(edits)


class _ProjectScripts:
    """The executable scripts a governing project ships: files under the
    manifest's directory with a shebang and the execute bit, outside hidden,
    virtual-environment and dependency directories. A script that runs a module
    by path or by `-m` names it in its text."""

    skipped_directories: ClassVar[frozenset[str]] = frozenset(
        ("node_modules", "__pycache__", "venv")
    )

    @classmethod
    def governing(cls, manifests: Iterable[Path]) -> tuple[Path, ...]:
        scripts: set[Path] = set()
        for manifest in manifests:
            for directory, subdirectories, files in os.walk(manifest.parent):
                subdirectories[:] = [
                    name
                    for name in subdirectories
                    if not name.startswith(".") and name not in cls.skipped_directories
                ]
                for name in files:
                    path = Path(directory) / name
                    if path.suffix != ".py" and os.access(path, os.X_OK) and cls._has_shebang(path):
                        scripts.add(path)
        return tuple(sorted(scripts))

    @staticmethod
    def _has_shebang(path: Path) -> bool:
        with path.open("rb") as script:
            return script.read(2) == b"#!"


class _ScriptCode:
    """Offsets of a script's code: everything but its comments. A `#` starts a
    comment at the start of a line or after whitespace, outside quotes; the
    shebang is one."""

    @staticmethod
    def spans(source: str) -> Iterator[tuple[int, int]]:
        start = index = 0
        quote = ""
        previous = "\n"
        while index < len(source):
            character = source[index]
            if quote:
                if character == "\\" and quote == '"':
                    index += 1
                elif character == quote:
                    quote = ""
            elif character in "\"'":
                quote = character
            elif character == "#" and previous.isspace():
                yield start, index
                newline = source.find("\n", index)
                index = start = len(source) if newline < 0 else newline
                previous = "\n"
                continue
            previous = character
            index += 1
        yield start, len(source)


class _TomlStrings:
    """Offsets of every string's text in a TOML document: basic and literal,
    single- and multi-line, skipping comments."""

    @staticmethod
    def content_spans(source: str) -> Iterator[tuple[int, int]]:
        index = 0
        while index < len(source):
            character = source[index]
            if character == "#":
                newline = source.find("\n", index)
                index = len(source) if newline < 0 else newline
            elif source.startswith(('"""', "'''"), index):
                delimiter = source[index : index + 3]
                end = _TomlStrings._closing(source, index + 3, delimiter)
                yield index + 3, end
                index = end + 3
            elif character in "\"'":
                end = _TomlStrings._closing(source, index + 1, character)
                yield index + 1, end
                index = end + 1
            else:
                index += 1

    @staticmethod
    def _closing(source: str, start: int, delimiter: str) -> int:
        index = start
        while not source.startswith(delimiter, index):
            if index >= len(source):
                raise ValueError(f"Unterminated TOML string at offset {start}")
            index += 2 if delimiter[0] == '"' and source[index] == "\\" else 1
        return index


@dataclass(frozen=True)
class _ModuleIdentityStrings:
    """Every mention of the dissolved module inside a string (docstrings and
    f-string parts included), in Python sources and in the `pyproject.toml`
    governing them: its dotted name, `<module>.<Name>` (a patch target),
    `<module>:<Name>` (an entry point), or its path. Each mention is rewritten
    in place to the name's owner or to the successor; a mention in the claim
    (first positional argument) of a call named in `retained_calls` (absence
    evidence such as PathsAbsent) stays."""

    source_path: str
    source_module_name: str
    owners: Mapping[str, str]
    successor_path: str | None
    retained_calls: frozenset[str] = frozenset()

    @property
    def module_pattern(self) -> re.Pattern[str]:
        return re.compile(
            rf"(?<![\w.]){re.escape(self.source_module_name)}(?:([.:])(\w+))?(?!\w)"
        )

    @property
    def path_pattern(self) -> re.Pattern[str]:
        file_name = self.source_path.rsplit("/", 1)[-1]
        return re.compile(rf"(?<![\w./-])((?:[\w.-]+/)+{re.escape(file_name)})(?!\w)")

    def string_replacements(
        self,
        context: CodemodSelectorContext,
        where: str,
        segment: str,
        start: int,
        routed: Iterable[tuple[int, int]] = (),
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Replacements for every mention of the module in one string's text,
        which starts at offset `start` of its file."""

        routed = tuple(routed)
        replacements: list[SourceTextSpanReplacement] = []
        for match in self.module_pattern.finditer(segment):
            if any(low <= match.start() < high for low, high in routed):
                continue
            replacement_source = self._module_replacement(context, where, match)
            if replacement_source == match.group(0):
                continue
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start + match.start(),
                    end_offset=start + match.end(),
                    replacement_source=replacement_source,
                )
            )
        for match in self.path_pattern.finditer(segment):
            for path in self._source_path_suffixes(match.group(1))[:1]:
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start + match.end(1) - len(path),
                        end_offset=start + match.end(1),
                        replacement_source=self._path_replacement(where, path),
                    )
                )
        return tuple(replacements)

    def _source_path_suffixes(self, spelled: str) -> tuple[str, ...]:
        """The tails of a spelled path, longest first, that name the dissolved
        module's file: `$root/src/pkg/worker.py` names `src/pkg/worker.py`. A
        tail keeps at least one directory, so a bare file name names nothing."""

        parts = spelled.split("/")
        tails = ("/".join(parts[index:]) for index in range(len(parts) - 1))
        return tuple(
            tail
            for tail in tails
            if self.source_path == tail or self.source_path.endswith(f"/{tail}")
        )

    def manifest_edits(
        self,
        context: CodemodSelectorContext,
        operation: RefactorRecipeOperation,
    ) -> tuple[TextFileEdit, ...]:
        """Every string in a governing `pyproject.toml` naming the module: a
        `[project.scripts]` or `[project.entry-points]` value, a tool's module
        or path setting."""

        edits: list[TextFileEdit] = []
        for manifest in _ProjectManifests.governing(context.execution_snapshot().sources_by_file_path):
            source = context.execution_snapshot().text_file(manifest).required_text(manifest)
            replacements = tuple(
                replacement
                for start, end in _TomlStrings.content_spans(source)
                for replacement in self.string_replacements(
                    context,
                    f"{manifest.as_posix()!r}:{source.count(chr(10), 0, start) + 1}",
                    source[start:end],
                    start,
                )
            )
            if not replacements:
                continue
            edit = TextFileEdit(
                operation_type=type(operation),
                file_path=manifest.as_posix(),
                source=source,
                replacements=replacements,
                rationale=f"Entry points and settings naming {self.source_module_name!r} follow it.",
            )
            tomllib.loads(edit.rewritten)
            edits.append(edit)
        return tuple(edits)

    def script_edits(
        self,
        context: CodemodSelectorContext,
        operation: RefactorRecipeOperation,
    ) -> tuple[TextFileEdit, ...]:
        """Every mention of the module in the code of an executable script the
        governing projects ship: `python -m module`, or its path after
        `$repo_root/`. Comments are prose and stay as written."""

        manifests = _ProjectManifests.governing(context.execution_snapshot().sources_by_file_path)
        edits: list[TextFileEdit] = []
        for script in _ProjectScripts.governing(manifests):
            source = context.execution_snapshot().text_file(script).required_text(script)
            replacements = tuple(
                replacement
                for start, end in _ScriptCode.spans(source)
                for replacement in self.string_replacements(
                    context,
                    f"{script.as_posix()!r}:{source.count(chr(10), 0, start) + 1}",
                    source[start:end],
                    start,
                )
            )
            if replacements:
                edits.append(
                    TextFileEdit(
                        operation_type=type(operation),
                        file_path=script.as_posix(),
                        source=source,
                        replacements=replacements,
                        rationale=f"Scripts running {self.source_module_name!r} follow it.",
                    )
                )
        return tuple(edits)

    def edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        embedded = _EmbeddedImportFrom(self.source_module_name)
        edits: list[NominalSourceEdit] = []
        for source_file, parsed in FilesReachingModule(context, self.source_module_name):
            file_path = source_file.file_path
            if file_path == self.source_path:
                continue
            source = parsed.source
            geometry = SourceTextGeometry(source)
            retained = self._retained_nodes(parsed.fact(ModuleNodeIndex))
            replacements: list[SourceTextSpanReplacement] = []
            for node in parsed.fact(ModuleNodeIndex).string_constants:
                if id(node) in retained:
                    continue
                offsets = geometry.node_offsets(node)
                if offsets is None:
                    continue
                start, end = offsets
                segment = source[start:end]
                routed: list[tuple[int, int]] = []
                for match in embedded.matches(segment):
                    unowned = [name for name in embedded.names(match) if name not in self.owners]
                    if unowned:
                        raise ValueError(
                            f"{file_path!r}:{node.lineno} imports {unowned!r} from the "
                            f"dissolved module {self.source_module_name!r} inside a "
                            "string; it does not bind them"
                        )
                    routed.append(match.span())
                replacements.extend(
                    self.string_replacements(
                        context, f"{file_path!r}:{node.lineno}", segment, start, routed
                    )
                )
            replacements.extend(
                self._glued_replacements(context, file_path, parsed.fact(ModuleNodeIndex), geometry)
            )
            if replacements:
                edits.append(geometry.nominal_edit(file_path=file_path, replacements=tuple(replacements)))
        return tuple(edits)

    def _glued_replacements(
        self,
        context: CodemodSelectorContext,
        file_path: str,
        index: ModuleNodeIndex,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """A module name glued together in an f-string, `f"import pkg.{module}"`,
        names the dissolved module through every literal bound to `module`: a
        `pytest.mark.parametrize` value or a `for` loop's literal iterable. Each
        literal spelling the module's remainder after the constant prefix follows
        the successor. A glue whose name NRA cannot trace, in a file that spells
        that remainder, refuses instead of leaving the string behind."""

        replacements: list[SourceTextSpanReplacement] = []
        for joined, prefix, name in _GluedModuleName.sites(index):
            if not self.source_module_name.startswith(prefix):
                continue
            remainder = self.source_module_name[len(prefix):]
            literals = _GluedModuleName.bound_literals(index, joined, name)
            if literals is None:
                if any(node.value == remainder for node in index.string_constants):
                    raise ValueError(
                        f"{file_path!r}:{joined.lineno} glues {prefix!r} onto {name!r}, "
                        f"which NRA cannot trace, and the file spells {remainder!r}; "
                        "bind the name from a parametrize or a literal for loop"
                    )
                continue
            for literal in literals:
                if literal.value != remainder:
                    continue
                if self.successor_path is None:
                    raise ValueError(
                        f"{file_path!r}:{literal.lineno} glues the dissolved module "
                        f"{self.source_module_name!r} together; declare its successor_path"
                    )
                successor = context.module_import_graph.module_name_for_file_path(
                    self.successor_path
                )
                if not successor.startswith(prefix):
                    raise ValueError(
                        f"{file_path!r}:{literal.lineno} glues {prefix!r} onto "
                        f"{remainder!r}; the successor {successor!r} is not under it"
                    )
                start, end = geometry.required_node_offsets(literal)
                quote = geometry.source[start]
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=f"{quote}{successor[len(prefix):]}{quote}",
                    )
                )
        return tuple(replacements)

    def _retained_nodes(self, index: ModuleNodeIndex) -> frozenset[int]:
        if not self.retained_calls:
            return frozenset()
        return frozenset(
            id(inner)
            for node in index.calls
            if (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None
            ) in self.retained_calls
            for argument in node.args[:1]
            for inner in ast.walk(argument)
        )

    def _module_replacement(
        self,
        context: CodemodSelectorContext,
        where: str,
        match: re.Match[str],
    ) -> str:
        import_graph = context.module_import_graph
        separator, name = match.group(1), match.group(2)
        if name is not None and name in self.owners:
            owner = import_graph.module_name_for_file_path(self.owners[name])
            return f"{owner}{separator}{name}"
        if name is not None:
            raise ValueError(
                f"{where} names {match.group(0)!r}, which the "
                "dissolved module does not bind"
            )
        if self.successor_path is None:
            raise ValueError(
                f"{where} names the dissolved module "
                f"{match.group(0)!r}; declare its successor_path or retain the call"
            )
        return import_graph.module_name_for_file_path(self.successor_path)

    def _path_replacement(self, where: str, path: str) -> str:
        if self.successor_path is None:
            raise ValueError(
                f"{where} names the dissolved module's path "
                f"{path!r}; declare its successor_path or retain the call"
            )
        root = self.source_path[: len(self.source_path) - len(path)]
        if not self.successor_path.startswith(root):
            raise ValueError(
                f"{where} names {path!r} relative to a root "
                f"{self.successor_path!r} does not share"
            )
        return self.successor_path[len(root):]


@dataclass(frozen=True)
class DissolvedModuleReport(DataclassJsonReport):
    """Where a dissolved module's names went."""

    deleted_path: str
    owner_paths_by_name: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _RelocatedNameStrings(_ModuleIdentityStrings):
    """A relocation's mentions of its source, which stays: only
    `<module>.<Name>` and `<module>:<Name>` of a name that left follow it to its
    owner. The module's own name, its path and the names it still declares
    name what is still there."""

    def _module_replacement(
        self,
        context: CodemodSelectorContext,
        where: str,
        match: re.Match[str],
    ) -> str:
        if match.group(2) not in self.owners:
            return match.group(0)
        return super()._module_replacement(context, where, match)

    def _source_path_suffixes(self, spelled: str) -> tuple[str, ...]:
        """The source file stays where it is, so no path to it changes."""

        del spelled
        return ()


@dataclass(frozen=True, kw_only=True)
class DissolveModuleOperation(RetainedCallOperation):
    """Delete a module that declares nothing: every name it still binds is
    imported by its consumers from the module that declares it (through
    `ModuleNameRedirect`), every binding of the module object that only read
    those names is removed, and every string naming the module follows it.

    `successor_path` names the module a string naming the dissolved module now
    means; without one such a string is refused. A bare import taken only to run
    the module runs the modules it imported instead. A consumer that binds the
    module otherwise for its import side effect, or imports a name the module took
    from outside the index, is refused: it has no owner to follow."""

    successor_path: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def text_file_changes(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[TextFileEdit, ...]:
        strings = self._identity_strings(context)
        return (*strings.manifest_edits(context, self), *strings.script_edits(context, self))

    def source_file_deletions(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileDeletion, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        return (
            SourceFileDeletion(
                operation_type=type(self),
                file_path=source_path,
                rationale=self.rationale_text(f"Dissolve {source_path!r}."),
            ),
        )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.required_reproof(lambda: self.source_edits_from_snapshot(context))
            source_path = self.required_source_path(context, self.operation_key())
            bindings = _DissolvedModuleBindings.of(context, source_path)
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Every consumer of the dissolved module follows its owners",
                detail=DissolvedModuleReport(
                    deleted_path=source_path,
                    owner_paths_by_name=tuple(sorted(bindings.owners.items())),
                ),
            ),
        )

    def _identity_strings(self, context: CodemodSelectorContext) -> _ModuleIdentityStrings:
        source_path = self.required_source_path(context, self.operation_key())
        source_module_name = context.module_import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(f"Source module identity is unavailable for {source_path!r}")
        return _ModuleIdentityStrings(
            source_path,
            source_module_name,
            _DissolvedModuleBindings.of(context, source_path).owners,
            (
                None
                if self.successor_path is None
                else SourcePathResolutionAuthority.from_source_index(
                    self.successor_path,
                    context.source_index,
                ).required_path()
            ),
            frozenset(self.retained_calls),
        )

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        strings = self._identity_strings(context)
        bindings = _DissolvedModuleBindings.of(context, strings.source_path)
        excluded = frozenset({strings.source_path})
        self._require_followable_consumers(context, excluded, strings.source_module_name, bindings)
        return (
            *self._module_binding_edits(
                context, excluded, strings.source_module_name, bindings.run_modules(context)
            ),
            *ModuleNameRedirect(strings.source_path, bindings.owners).consumer_edits(
                context,
                excluded_paths=excluded,
            ),
            *strings.edits(context),
            *(
                edit
                for entry in _ScriptEntry.of(context.module_nodes_by_file_path[strings.source_path])
                for edit in entry.successor_edits(context, strings.source_path, strings.successor_path)
            ),
        )

    @staticmethod
    def _require_followable_consumers(
        context: CodemodSelectorContext,
        excluded_paths: frozenset[str],
        source_module_name: str,
        bindings: _DissolvedModuleBindings,
    ) -> None:
        for source_file, parsed in FilesReachingModule(context, source_module_name):
            if source_file.file_path in excluded_paths:
                continue
            identity = source_file.module_path_identity
            for node in parsed.fact(ModuleNodeIndex).import_froms():
                if (
                    identity.resolve_import_from_module(
                        imported_module=node.module,
                        level=node.level,
                    )
                    != source_module_name
                ):
                    continue
                unowned = tuple(
                    alias.name
                    for alias in node.names
                    if alias.name == "*" and bindings.foreign_names
                    or alias.name != "*" and alias.name not in bindings.owners
                )
                if unowned:
                    raise ValueError(
                        f"{source_file.file_path!r}:{node.lineno} imports {unowned!r} "
                        f"from {source_module_name!r}, which declares none of them"
                    )

    @staticmethod
    def _module_binding_edits(
        context: CodemodSelectorContext,
        excluded_paths: frozenset[str],
        source_module_name: str,
        run_modules: tuple[str, ...],
    ) -> tuple[NominalSourceEdit, ...]:
        """Each bare `import pkg.source` taken only to run the module imports, in
        its place, the modules the dissolved module still runs (`run_modules`),
        when each binds the same package head; a relocation already added the
        modules its declarations went to. Refuse the bindings of the dissolved
        module that neither this nor its redirect can remove: a bare import whose
        dotted path is read, or whose run modules bind another head; one of a
        submodule; another binding taken only for its import side effect; and one
        inside a function. `ModuleNameRedirect` removes every other binding
        (`from pkg import source`, `import pkg.source as alias`) once its reads are
        redirected."""

        package, _, leaf = source_module_name.rpartition(".")
        head = source_module_name.partition(".")[0]
        other_heads = tuple(module for module in run_modules if module.partition(".")[0] != head)
        edits: list[NominalSourceEdit] = []
        for source_file, parsed in FilesReachingModule(context, source_module_name):
            if source_file.file_path in excluded_paths:
                continue
            identity = source_file.module_path_identity
            scoped = {
                id(statement): scope
                for scope in ModuleImportScope
                for statement in scope.import_statements(parsed.module)
            }
            reads_path = _RunOnlyImports.reads_path(parsed.module, source_module_name)
            for statement in parsed.fact(ModuleNodeIndex).imports:
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        if alias.name.startswith(f"{source_module_name}."):
                            raise ValueError(
                                f"{source_file.file_path!r}:{statement.lineno} imports "
                                f"a submodule of {source_module_name!r}"
                            )
                        if alias.asname is not None or alias.name != source_module_name:
                            continue
                        if reads_path:
                            raise ValueError(
                                f"{source_file.file_path!r}:{statement.lineno} imports "
                                f"{source_module_name!r} as a module path and reads through it"
                            )
                        if other_heads:
                            raise ValueError(
                                f"{source_file.file_path!r}:{statement.lineno} imports "
                                f"{source_module_name!r} to run it, which runs "
                                f"{other_heads!r}, outside {head!r}"
                            )
                    bound = tuple(
                        alias.asname
                        for alias in statement.names
                        if alias.asname is not None and alias.name == source_module_name
                    )
                elif isinstance(statement, ast.ImportFrom) and (
                    identity.resolve_import_from_module(
                        imported_module=statement.module,
                        level=statement.level,
                    )
                    == package
                ):
                    bound = tuple(
                        ImportBoundNameProjection(statement).alias_bound_name(alias)
                        for alias in statement.names
                        if alias.name == leaf
                    )
                else:
                    continue
                for name in bound:
                    if name not in parsed.fact(ModuleNodeIndex).attribute_bases:
                        raise ValueError(
                            f"{source_file.file_path!r}:{statement.lineno} binds "
                            f"{source_module_name!r} for its import side effect"
                        )
                if not bound:
                    continue
                scope = scoped.get(id(statement))
                if scope is None:
                    raise ValueError(
                        f"{source_file.file_path!r}:{statement.lineno} binds "
                        f"{source_module_name!r} inside a function"
                    )
            run_only = _RunOnlyImports.of(parsed, source_file.file_path, source_module_name)
            if run_only.statements:
                edits.append(
                    run_only.edit(
                        context.sources_by_file_path[source_file.file_path],
                        source_module_name,
                        run_modules,
                    )
                )
        return tuple(edits)


@dataclass(frozen=True, kw_only=True)
class RetargetModuleMentionsOperation(RetainedCallOperation):
    """Point every string mention of a module that no longer exists at its
    successor: the cleanup a dissolve does, for a module removed before the
    dissolve could see the mention (prose, embedded program text)."""

    retired_module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    retired_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    successor_path: str = codemod_payload_field(SourcePathPayloadValueCodec())

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.required_reproof(lambda: self.source_edits_from_snapshot(context))
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Every mention of the retired module names its successor",
                detail=DissolvedModuleReport(deleted_path=self.retired_path, owner_paths_by_name=()),
            ),
        )

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        if self.retired_module_name in context.module_import_graph.known_module_names:
            raise ValueError(
                f"{self.retired_module_name!r} still exists; dissolve it instead"
            )
        successor_path = SourcePathResolutionAuthority.from_source_index(
            self.successor_path,
            context.source_index,
        ).required_path()
        return _ModuleIdentityStrings(
            self.retired_path,
            self.retired_module_name,
            {},
            successor_path,
            frozenset(self.retained_calls),
        ).edits(context)


_RELATIVE_FROM = re.compile(r"from\s+(\.+)([\w.]*)(?=\s+import\b)")


@dataclass(frozen=True)
class _RelativeImportRebase:
    """A moved block's relative imports, re-expressed for the destination package.

    Top-level imports the block needs are re-derived by the move's dependency
    report; a relative import inside the block's body (a function-local
    ``from .x import Y``) moves with the text, so when the destination sits in
    another package it is re-resolved against the source package and written
    relative to the destination's."""

    source_identity: PythonModulePathIdentity
    destination_package: tuple[str, ...]

    @classmethod
    def between(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        destination_path: str,
    ) -> _RelativeImportRebase:
        import_graph = context.module_import_graph
        source_file = import_graph.source_file_for_path(source_path)
        destination_file = import_graph.source_file_for_path(destination_path)
        if source_file is None or destination_file is None:
            raise ValueError("A move needs both modules' identities to rebase relative imports")
        destination_module = destination_file.module_name
        package = (
            destination_module
            if destination_file.is_package_init
            else destination_module.rpartition(".")[0]
        )
        return cls(source_file.module_path_identity, tuple(filter(None, package.split("."))))

    def applied(self, block: str) -> str:
        tree = ast.parse(block)
        geometry = SourceTextGeometry(block)
        replacements: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level):
                continue
            start, _end = geometry.required_node_offsets(node)
            match = _RELATIVE_FROM.match(block, start)
            if match is None:
                raise ValueError(f"Cannot read the relative import at block line {node.lineno}")
            absolute = self.source_identity.resolve_import_from_module(
                imported_module=node.module,
                level=node.level,
            )
            if absolute is None:
                raise ValueError(f"Relative import at block line {node.lineno} leaves the package")
            replacements.append((start, match.end(), f"from {self._relative(absolute)}"))
        for start, end, text in sorted(replacements, reverse=True):
            block = block[:start] + text + block[end:]
        return block

    def _relative(self, absolute: str) -> str:
        target = tuple(absolute.split("."))
        common = 0
        while (
            common < min(len(target), len(self.destination_package))
            and target[common] == self.destination_package[common]
        ):
            common += 1
        if common == 0:
            return absolute
        dots = "." * (len(self.destination_package) - common + 1)
        return dots + ".".join(target[common:])
