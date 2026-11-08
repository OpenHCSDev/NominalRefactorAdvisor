from __future__ import annotations

import ast
import re
from abc import (
    ABC,
    abstractmethod,
)
from collections import Counter
from collections.abc import (
    Iterator,
    Mapping,
)
from dataclasses import dataclass
from itertools import product
from typing import ClassVar

from ..ast_tools import ModuleAnnotationEvaluationMode, ParsedModule
from ..module_node_index import ModuleNodeIndex
from ..codemod_import_bindings import FromModuleImportBindingIdentity
from ..codemod_import_graph import SourceModuleImportGraph
from ..source_index import SourceFileDigest
from ..codemod_import_scopes import ModuleImportScope
from ..codemod_imports import (
    ImportFromSource,
    ModuleImportMutation,
)
from ..codemod_module_declarations import (
    ModuleSymbolTable,
    MovedTopLevelDeclarationSource,
    SourceTopLevelSymbolMoveSelection,
)
from ..codemod_module_move_reports import (
    ModuleMoveDependencyReport,
    ModuleMoveImportDependency,
    ModuleMoveObstacle,
    ModuleMoveObstacleKind,
    ModuleMoveSourceLocalDependency,
    ModuleMoveSourceLocalDependencyResolution,
)
from ..codemod_operations import RefactorRecipeOperation
from ..codemod_paths import SourcePathResolutionAuthority
from ..codemod_payload import (
    OptionalStringPayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from ..codemod_reproof import RepositorySourceReprovedOperation
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourcePathPayloadValueCodec
from ..codemod_semantics import (
    CodemodPreflightStatus,
    CodemodSourceDependencyScope,
)
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceFileDeletion,
    SourceInsertion,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from ..codemod_spacing import DestinationInsertionSpacing
from ..declaration_dependencies import DeclarationDependencyProjection
from ..json_reports import DataclassJsonReport
from ..lexical_bindings import ImportFromModuleName
from ..public_exports import (
    CompactModulePublicExportContract,
    module_public_export_contract,
)
from ..python_module_identity import PythonModulePathIdentity


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
        source_module_name = import_graph.module_name_for_file_path(self.source_path)
        if source_module_name is None:
            raise ValueError(
                f"Source module identity is unavailable for {self.source_path!r}"
            )
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
                owner_modules={
                    name: owner
                    for name, path in self.destinations.items()
                    if (owner := import_graph.module_name_for_file_path(path)) is not None
                },
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
            local_imports = self._in_place_import_replacements(
                import_graph,
                parsed,
                identity,
                consumer_path,
                source_module_name,
            )
            import_strings = self._import_string_replacements(
                import_graph,
                parsed,
                source_module_name,
            )
            if import_strings:
                edits.append(
                    SourceTextGeometry(context.sources_by_file_path[consumer_path]).nominal_edit(
                        file_path=consumer_path,
                        replacements=import_strings,
                    )
                )
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
                    ).source.strip()
                )
            replacements.append(
                (statement, ("\n" + " " * statement.col_offset).join(lines))
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
                ).source.strip()
                for destination_path in sorted({self.destinations[alias.name] for alias in moved})
            ]
            if kept:
                lines.append(ImportFromSource(module_name=source_module_name, aliases=kept).source.strip())
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

    def patched_reader_path(self, name: str, file_path: str) -> str:
        """The module whose code reads `name` once it leaves the source, which a
        patch of `source.name` must target instead. These owners spread the
        readers, so no one module is it."""

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
                ).source,
                scope=scope,
            )


@dataclass(frozen=True)
class RelocationNameRedirect(ModuleNameRedirect):
    """The redirect of one relocation: every moved declaration, and so every
    reader of a removed source import, lands in `reader_path`."""

    reader_path: str

    def patched_reader_path(self, name: str, file_path: str) -> str:
        return self.reader_path


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


@dataclass(frozen=True)
class _NamedModuleMemberRead(_NamedModuleMemberAccess):
    """`getattr`/`hasattr` reads the name where it is now declared."""

    ACCESSORS: ClassVar[frozenset[str]] = frozenset(
        accessor.__name__ for accessor in (getattr, hasattr)
    )

    def target_path(self, redirect: ModuleNameRedirect, file_path: str) -> str:
        return redirect.destinations[self.member_name]


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


@dataclass(frozen=True)
class _SourceModuleAliasRetargets:
    """`import source as s` whose every load of `s` reads a name now declared in
    one module binds that module where it stands: the same scope, the same alias,
    the reads as written. A function that imports the module to keep it off the
    import path at load time keeps doing so. An alias that also reads a name
    staying behind, reaches names moved to several owners, or is used any other
    way stays with `_SourceModuleObjectReads`."""

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
        owner_modules: Mapping[str, str],
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
                for owner in cls._sole_owner(index, alias.asname, scope, owner_modules)
            )
        )

    @staticmethod
    def _sole_owner(
        index: ModuleNodeIndex,
        binding: str,
        scope: ast.AST | None,
        owner_modules: Mapping[str, str],
    ) -> tuple[str, ...]:
        """The one module every use of the alias's variable in `scope` reads a
        moved name from, or none."""

        owners: set[str] = set()
        for node in index.uses_of(binding, scope):
            parent = index.parents.get(node)
            if not (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in owner_modules
            ):
                return ()
            owners.add(owner_modules[parent.attr])
        return tuple(owners) if len(owners) == 1 else ()

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
            (alias.asname or alias.name.split(".")[0], scope)
            for scope in ModuleImportScope
            for statement in scope.import_statements(parsed.module)
            for alias in statement.names
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
                        yield (alias.asname or alias.name, *source_parts[len(alias_parts) :])

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
            context.module_import_graph,
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
                owner_modules={name: destination_module_name for name in moved_symbol_names},
            ).localized_edits(
                destination_index,
                file_path=request.destination_path,
                source=context.sources_by_file_path[request.destination_path],
                reader_module=destination_module_name,
            )
        )
        consumer_edits = (
            *destination_localizations,
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
            *RelocationNameRedirect(
                source_path=request.source_path,
                destinations={
                    **cls._removed_binding_owners(context, report),
                    **{name: request.destination_path for name in moved_symbol_names},
                },
                reader_path=request.destination_path,
            ).consumer_edits(
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
            consumer_edits=consumer_edits,
            rationale=request.rationale,
        )

    @staticmethod
    def _removed_binding_owners(
        context: CodemodSelectorContext,
        report: ModuleMoveDependencyReport,
    ) -> dict[str, str]:
        """Source imports the move removes, each to the module declaring it, so a
        reader that reached the name through the source reads it from its owner.
        A name declared outside the index has no owner path and is left alone."""

        return {
            dependency.name: owner
            for dependency in report.source_removal_dependencies
            if dependency.identity.imported_name == dependency.name
            if (
                owner := ImportedNameOwner.declaring_path(
                    context,
                    dependency.identity.module_name,
                    dependency.name,
                )
            )
            is not None
        }

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
        import_graph: SourceModuleImportGraph,
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
        source_module_context_names = tuple(
            sorted(
                moved_dependencies.names
                & source_table.implicit_module_names
                - operation.module_context_names_following(source_table, moved_names)
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
                    source_module_context_names,
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
            rebase.applied(block.moved_source.strip("\n")) for block in self.source_blocks
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
    """A module operation that rewrites strings naming the module, except under
    a call named in `retained_calls`: those strings record the module's old
    identity on purpose (evidence that the path is gone, such as PathsAbsent)
    inside an otherwise live file, and stay as written."""

    retained_calls: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RepositorySourceReprovedOperation, ABC):
    """Repository-proved destination contract for module-symbol moves."""

    destination_path: str = codemod_payload_field(SourcePathPayloadValueCodec())

    def module_context_names_following(
        self, source_table: ModuleSymbolTable, moved_names: frozenset[str]
    ) -> frozenset[str]:
        """Module-context names whose meaning moves with the code. Into a module that
        already exists, none: `__name__` there names another module."""
        return frozenset()

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


@dataclass(frozen=True, kw_only=True)
class NewModuleSymbolMoveOperationABC(ModuleSymbolMoveOperation, ABC):
    """Module move whose destination source is created atomically."""

    def module_context_names_following(
        self, source_table: ModuleSymbolTable, moved_names: frozenset[str]
    ) -> frozenset[str]:
        """When every name the source declares moves into the module this move creates,
        the moved code is that module under its new name, so `__name__` (a logger's, say)
        names the destination. `__file__` never follows: a path beside the module stays."""
        if source_table.declared_names <= moved_names:
            return frozenset(("__name__",))
        return frozenset()

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
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=self.destination_path,
                source_index=context.source_index,
                source=self.initial_destination_source(context),
            ),
        )

    def initial_destination_source(self, context: CodemodSelectorContext) -> str:
        """Resolve caller source, or the caller's module docstring followed by the
        source module's derived annotation policy."""

        if self.destination_source is not None:
            return self.destination_source
        source_path = self.required_source_path(context, self.operation_key())
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(
            context.module_nodes_by_file_path[source_path]
        )
        if self.destination_docstring is None:
            return annotation_mode.new_module_prelude()
        return f'"""{self.destination_docstring}"""\n\n{annotation_mode.new_module_prelude()}'

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

    @staticmethod
    def _binds_only_imports(statement: ast.stmt, *, first: bool) -> bool:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
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
                        yield alias.asname or alias.name.split(".")[0], None
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
        its binding is not a parametrize value or a literal loop iterable."""

        while (node := index.parents.get(node)) is not None:
            match node:
                case ast.For(target=ast.Name(id=target), iter=iterable) if target == name:
                    return cls._literals(iterable)
                case ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
                    for generator in node.generators:
                        match generator.target:
                            case ast.Name(id=target) if target == name:
                                return cls._literals(generator.iter)
                case ast.Lambda(args=arguments) if name in {a.arg for a in arguments.args}:
                    return None
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ) if name in {argument.arg for argument in arguments.args}:
                    for decorator in node.decorator_list:
                        literals = cls._parametrized(decorator, name)
                        if literals is not None:
                            return literals
                    return None
        return None

    @classmethod
    def _parametrized(cls, decorator: ast.expr, name: str) -> tuple[ast.Constant, ...] | None:
        match decorator:
            case ast.Call(
                func=ast.Attribute(attr="parametrize"),
                args=[ast.Constant(value=str(argnames)), ast.Tuple(elts=rows) | ast.List(elts=rows), *_],
            ):
                names = [part.strip() for part in argnames.split(",")]
            case _:
                return None
        if name not in names:
            return None
        column = names.index(name)
        cells: list[ast.expr] = []
        for row in rows:
            match row:
                case ast.Call(func=ast.Attribute(attr="param"), args=values):
                    pass
                case ast.Tuple(elts=values) | ast.List(elts=values) if len(names) > 1:
                    pass
                case _:
                    values = [row]
            if len(values) <= column:
                return None
            cells.append(values[column])
        return cls._literals(ast.Tuple(elts=cells))

    @staticmethod
    def _literals(iterable: ast.expr) -> tuple[ast.Constant, ...] | None:
        match iterable:
            case ast.Tuple(elts=elements) | ast.List(elts=elements) | ast.Set(elts=elements):
                pass
            case _:
                return None
        literals: list[ast.Constant] = []
        for element in elements:
            match element:
                case ast.Constant(value=str()):
                    literals.append(element)
                case _:
                    return None
        return tuple(literals)


@dataclass(frozen=True)
class _ModuleIdentityStrings:
    """Every mention of the dissolved module inside a string (docstrings and
    f-string parts included): its dotted name, `<module>.<Name>` (a patch
    target), or its path. Each mention is rewritten in place to the name's
    owner or to the successor; a mention under a call named in
    `retained_calls` (absence evidence such as PathsAbsent) stays."""

    source_path: str
    source_module_name: str
    owners: Mapping[str, str]
    successor_path: str | None
    retained_calls: frozenset[str] = frozenset()

    def edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        module_pattern = re.compile(
            rf"(?<![\w.]){re.escape(self.source_module_name)}(?:\.(\w+))?(?!\w)"
        )
        embedded = _EmbeddedImportFrom(self.source_module_name)
        file_name = self.source_path.rsplit("/", 1)[-1]
        path_pattern = re.compile(rf"(?<![\w./-])((?:[\w.-]+/)+{re.escape(file_name)})(?!\w)")
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
                for match in module_pattern.finditer(segment):
                    if any(low <= match.start() < high for low, high in routed):
                        continue
                    replacement = self._module_replacement(context, file_path, node, match)
                    replacements.append(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start + match.start(),
                            end_offset=start + match.end(),
                            replacement_source=replacement,
                        )
                    )
                for match in path_pattern.finditer(segment):
                    path = match.group(1)
                    if not (self.source_path == path or self.source_path.endswith(f"/{path}")):
                        continue
                    replacements.append(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start + match.start(1),
                            end_offset=start + match.end(1),
                            replacement_source=self._path_replacement(file_path, node, path),
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
            for argument in (*node.args, *(keyword.value for keyword in node.keywords))
            for inner in ast.walk(argument)
        )

    def _module_replacement(
        self,
        context: CodemodSelectorContext,
        file_path: str,
        node: ast.Constant,
        match: re.Match[str],
    ) -> str:
        import_graph = context.module_import_graph
        name = match.group(1)
        if name is not None and name in self.owners:
            return f"{import_graph.module_name_for_file_path(self.owners[name])}.{name}"
        if name is not None:
            raise ValueError(
                f"{file_path!r}:{node.lineno} names {match.group(0)!r}, which the "
                "dissolved module does not bind"
            )
        if self.successor_path is None:
            raise ValueError(
                f"{file_path!r}:{node.lineno} names the dissolved module "
                f"{match.group(0)!r}; declare its successor_path or retain the call"
            )
        return import_graph.module_name_for_file_path(self.successor_path)

    def _path_replacement(self, file_path: str, node: ast.Constant, path: str) -> str:
        if self.successor_path is None:
            raise ValueError(
                f"{file_path!r}:{node.lineno} names the dissolved module's path "
                f"{path!r}; declare its successor_path or retain the call"
            )
        root = self.source_path[: len(self.source_path) - len(path)]
        if not self.successor_path.startswith(root):
            raise ValueError(
                f"{file_path!r}:{node.lineno} names {path!r} relative to a root "
                f"{self.successor_path!r} does not share"
            )
        return self.successor_path[len(root):]


@dataclass(frozen=True)
class DissolvedModuleReport(DataclassJsonReport):
    """Where a dissolved module's names went."""

    deleted_path: str
    owner_paths_by_name: tuple[tuple[str, str], ...]


@dataclass(frozen=True, kw_only=True)
class DissolveModuleOperation(RetainedCallOperation):
    """Delete a module that declares nothing: every name it still binds is
    imported by its consumers from the module that declares it (through
    `ModuleNameRedirect`), every binding of the module object that only read
    those names is removed, and every string naming the module follows it.

    `successor_path` names the module a string naming the dissolved module now
    means; without one such a string is refused. A consumer that binds the
    module for its import side effect, or imports a name the module took from
    outside the index, is refused: it has no owner to follow."""

    successor_path: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

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

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        import_graph = context.module_import_graph
        source_module_name = import_graph.module_name_for_file_path(source_path)
        if source_module_name is None:
            raise ValueError(f"Source module identity is unavailable for {source_path!r}")
        bindings = _DissolvedModuleBindings.of(context, source_path)
        successor_path = (
            None
            if self.successor_path is None
            else SourcePathResolutionAuthority.from_source_index(
                self.successor_path,
                context.source_index,
            ).required_path()
        )
        excluded = frozenset({source_path})
        self._require_followable_consumers(context, excluded, source_module_name, bindings)
        self._require_removable_module_bindings(context, excluded, source_module_name)
        return (
            *ModuleNameRedirect(source_path, bindings.owners).consumer_edits(
                context,
                excluded_paths=excluded,
            ),
            *_ModuleIdentityStrings(
                source_path,
                source_module_name,
                bindings.owners,
                successor_path,
                frozenset(self.retained_calls),
            ).edits(context),
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
    def _require_removable_module_bindings(
        context: CodemodSelectorContext,
        excluded_paths: frozenset[str],
        source_module_name: str,
    ) -> None:
        """Refuse the bindings of the dissolved module that its redirect cannot
        remove: a bare `import pkg.source` (it binds the package head, which other
        reads may still need), a binding taken only for its import side effect,
        and one inside a function. `ModuleNameRedirect` removes every other
        binding (`from pkg import source`, `import pkg.source as alias`) once its
        reads are redirected."""

        package, _, leaf = source_module_name.rpartition(".")
        for source_file, parsed in FilesReachingModule(context, source_module_name):
            if source_file.file_path in excluded_paths:
                continue
            identity = source_file.module_path_identity
            scoped = {
                id(statement): scope
                for scope in ModuleImportScope
                for statement in scope.import_statements(parsed.module)
            }
            for statement in parsed.fact(ModuleNodeIndex).imports:
                if isinstance(statement, ast.Import):
                    if any(
                        alias.asname is None and alias.name == source_module_name
                        or alias.name.startswith(f"{source_module_name}.")
                        for alias in statement.names
                    ):
                        raise ValueError(
                            f"{source_file.file_path!r}:{statement.lineno} imports "
                            f"{source_module_name!r} as a module path"
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
                        alias.asname or alias.name
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
