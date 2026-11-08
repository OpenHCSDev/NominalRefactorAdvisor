"""Dependency-closed source movement codemods."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

from .ast_tools import ModuleAnnotationEvaluationMode
from .json_reports import DataclassJsonReport
from .class_index import CompactModulePublicExportContract, module_public_export_contract
from .codemod_import_bindings import FromModuleImportBindingIdentity
from .codemod_import_graph import SourceModuleImportGraph
from .codemod_import_scopes import ModuleImportScope
from .codemod_imports import (
    ImportFromSource,
    ModuleImportMutation,
)
from .codemod_module_declarations import (
    ModuleSymbolTable,
    MovedTopLevelDeclarationSource,
    SourceTopLevelSymbolMoveSelection,
)
from .codemod_module_move_reports import (
    ModuleMoveDependencyReport,
    ModuleMoveImportDependency,
    ModuleMoveObstacle,
    ModuleMoveObstacleKind,
    ModuleMoveSourceLocalDependency,
    ModuleMoveSourceLocalDependencyResolution,
)
from .codemod_paths import SourcePathResolutionAuthority
from .codemod_payload import (
    OptionalStringPayloadValueCodec,
    RequiredIntegerPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourcePathPayloadValueCodec
from .codemod_semantics import CodemodPreflightStatus
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceFileCreation,
    SourceFileDeletion,
    SourceInsertion,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from .codemod_spacing import DestinationInsertionSpacing
from .declaration_dependencies import DeclarationDependencyProjection
from .lexical_bindings import ImportFromModuleName
from .python_module_identity import PythonModulePathIdentity

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
        for source_file in context.source_index.files:
            consumer_path = source_file.file_path
            module = context.module_nodes_by_file_path.get(consumer_path)
            if consumer_path in excluded_paths or module is None:
                continue
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
                    named_aliases = tuple(
                        alias for alias in statement.names if alias.name in moved_names
                    )
                    if named_aliases:
                        edits.append(
                            ModuleImportMutation.remove_names(
                                file_path=consumer_path,
                                module_name=ImportFromModuleName.from_node(
                                    statement
                                ).source,
                                names=(alias.name for alias in named_aliases),
                                scope=scope,
                            )
                        )
                    edits.extend(
                        self._owner_imports(
                            import_graph,
                            consumer_path,
                            scope,
                            (
                                *named_aliases,
                                *(
                                    ast.alias(name=name, asname=name)
                                    for alias in statement.names
                                    if alias.name == "*"
                                    for name in star_exported.required(consumer_path)
                                ),
                            ),
                        )
                    )
            module_reads = _SourceModuleObjectReads.of(
                module,
                identity,
                source_module_name=source_module_name,
                moved_names=moved_names,
                file_path=consumer_path,
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
                edits.append(
                    module_reads.rewrite(context.sources_by_file_path[consumer_path])
                )
            local_imports = self._local_import_replacements(
                import_graph,
                module,
                identity,
                consumer_path,
                source_module_name,
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

    def _local_import_replacements(
        self,
        import_graph: SourceModuleImportGraph,
        module: ast.Module,
        identity: PythonModulePathIdentity,
        consumer_path: str,
        source_module_name: str,
    ) -> tuple[tuple[ast.ImportFrom, str], ...]:
        """Imports of moved names inside a function or class body, each rewritten in
        place: one statement per owner, plus one for the names that stay."""

        scoped = frozenset(
            id(statement)
            for scope in ModuleImportScope
            for statement in scope.import_statements(module)
        )
        replacements: list[tuple[ast.ImportFrom, str]] = []
        for statement in ast.walk(module):
            if (
                not isinstance(statement, ast.ImportFrom)
                or id(statement) in scoped
                or identity.resolve_import_from_module(
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
            lines = [
                ImportFromSource(
                    module_name=ModuleImportScope.RUNTIME.required_module_reference(
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
class _NamedModuleMemberAccess:
    """`getattr(m, "Name")`-shaped access: the module object is the first argument
    of a by-name accessor and the member is a string literal, so the name it
    reaches is as static as `m.Name`."""

    accessor: str
    member_name: str

    BUILTIN_ACCESSORS: ClassVar[frozenset[str]] = frozenset(
        accessor.__name__ for accessor in (getattr, hasattr, setattr, delattr)
    )
    PATCH_ACCESSORS: ClassVar[frozenset[str]] = frozenset(
        accessor.__name__ for accessor in (setattr, delattr)
    )
    PATCH_FIXTURE = "monkeypatch"

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
        if accessor in cls.BUILTIN_ACCESSORS or accessor in {
            f"{cls.PATCH_FIXTURE}.{name}" for name in cls.PATCH_ACCESSORS
        }:
            return cls(accessor=accessor, member_name=member.value)
        return None


@dataclass(frozen=True)
class _SourceModuleObjectReads:
    """A consumer's `module.Name` reads of moved names through its bindings of the
    source module object. The move refuses when such a binding is used other than
    by attribute reads, is rebound, or when a read name is already bound there."""

    file_path: str
    reads: tuple[tuple[ast.Attribute, str], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted({name for _node, name in self.reads}))

    @classmethod
    def of(
        cls,
        module: ast.Module,
        identity: PythonModulePathIdentity,
        *,
        source_module_name: str,
        moved_names: frozenset[str],
        file_path: str,
    ) -> _SourceModuleObjectReads:
        source_parts = tuple(source_module_name.split("."))
        chains = frozenset(cls._bound_chains(module, identity, source_parts))
        if not chains:
            return cls(file_path, ())
        bindings = cls._bound_name_counts(module)
        parents = {
            child: parent
            for parent in ast.walk(module)
            for child in ast.iter_child_nodes(parent)
        }
        reads: list[tuple[ast.Attribute, str]] = []
        for chain in chains:
            if bindings[chain[0]] > 1:
                raise ValueError(
                    f"{file_path!r} rebinds {chain[0]!r}, which binds the source "
                    f"module {source_module_name!r}"
                )
            for node in ast.walk(module):
                if not (isinstance(node, ast.Name) and node.id == chain[0]):
                    continue
                outermost, parts = cls._read_chain(node, parents, len(chain) + 1)
                if parts[: len(chain)] == chain and len(parts) > len(chain):
                    if parts[-1] in moved_names:
                        reads.append((outermost, parts[-1]))
                elif chain[: len(parts)] == parts and (
                    named := _NamedModuleMemberAccess.of(outermost, parents)
                ):
                    reached = (*parts, named.member_name)
                    if reached == chain or (
                        parts == chain and named.member_name in moved_names
                    ):
                        raise ValueError(
                            f"{file_path!r} names {'.'.join(reached)!r}, which "
                            f"reaches the moved names, through {named.accessor!r}"
                        )
                elif parts == chain:
                    raise ValueError(
                        f"{file_path!r} uses the source module object "
                        f"{'.'.join(parts)!r} other than by reading an attribute"
                    )
        taken = tuple(sorted({name for _node, name in reads if bindings[name]}))
        if taken:
            raise ValueError(
                f"{file_path!r} already binds {taken!r}, which its source-module "
                "reads would be redirected to"
            )
        return cls(file_path, tuple(reads))

    @staticmethod
    def _bound_chains(
        module: ast.Module,
        identity: PythonModulePathIdentity,
        source_parts: tuple[str, ...],
    ) -> Iterator[tuple[str, ...]]:
        """The attribute chain that reaches the source module from each binding."""

        for node in ast.walk(module):
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
    def _bound_name_counts(module: ast.Module) -> Counter[str]:
        """How many bindings each name has. Every `import pkg.x` binds the same
        package object to `pkg`, so those count as one binding however many."""

        counts: Counter[str] = Counter()
        package_heads: set[str] = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                counts[node.id] += 1
            elif isinstance(node, ast.arg):
                counts[node.arg] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                counts[node.name] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname is None:
                        package_heads.add(alias.name.split(".")[0])
                    else:
                        counts[alias.asname] += 1
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        counts[alias.asname or alias.name] += 1
        counts.update(package_heads)
        return counts

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

    def rewrite(self, source: str) -> NominalSourceEdit:
        geometry = SourceTextGeometry(source)
        return geometry.nominal_edit(
            file_path=self.file_path,
            replacements=tuple(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=end,
                    replacement_source=name,
                )
                for node, name in self.reads
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
        )
        source_binding_import_sources = operation.source_binding_import_sources(
            context,
            source_table=source_table,
            source_path=request.source_path,
            destination_path=request.destination_path,
            moved_symbol_names=moved_symbol_names,
        )
        consumer_edits = (
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
            *ModuleNameRedirect(
                source_path=request.source_path,
                destinations={name: request.destination_path for name in moved_symbol_names},
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

    @classmethod
    def _dependency_report(
        cls,
        import_graph: SourceModuleImportGraph,
        source_table: ModuleSymbolTable,
        destination_table: ModuleSymbolTable,
        request: SourceTopLevelSymbolClosureMoveRequest,
        operation: ModuleSymbolMoveOperation,
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
            sorted(moved_dependencies.names & source_table.implicit_module_names)
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
                    scope=dependency.scope,
                    rationale=self.rationale
                    or (
                        "Remove imports used only by moved symbols "
                        f"{self.dependency_report.moved_symbol_names!r}."
                    ),
                )
            )
            for dependency in self.dependency_report.source_removal_dependencies
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
                self.destination_source(destination_source, insertion_line)
            ),
            leading_boundary=leading_boundary,
            rationale=self.rationale
            or (
                f"Move symbols {self.dependency_report.moved_symbol_names!r} "
                f"into {self.destination_path!r}."
            ),
        )

    def destination_source(self, destination_source: str, insertion_line: int) -> str:
        moved_source = "\n\n\n".join(
            block.moved_source.strip("\n") for block in self.source_blocks
        )
        spacing = DestinationInsertionSpacing.from_source(
            destination_source,
            insertion_line,
            inserted_source_is_import_block=False,
            boundary=self.source_blocks[0].declaration.destination_boundary,
        )
        return f"{spacing.leading_separator}{moved_source}{spacing.trailing_separator}"


@dataclass(frozen=True, kw_only=True)
class ModuleSymbolMoveOperation(RepositorySourceReprovedOperation, ABC):
    """Repository-proved destination contract for module-symbol moves."""

    destination_path: str = codemod_payload_field(SourcePathPayloadValueCodec())

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
        del destination_path
        export_contract = module_public_export_contract(
            context.parsed_module_for_source_path(source_path)
        )
        blocked_names = tuple(
            symbol_name
            for symbol_name in moved_symbol_names
            if not export_contract.allows_binding_relocation(symbol_name)
        )
        if blocked_names:
            raise ValueError(
                "Source export contract does not prove binding relocation for "
                f"{blocked_names!r}"
            )
        retained_names = source_table.referenced_names_excluding(
            moved_symbol_names,
            moved_symbol_names,
        ).intersection(moved_symbol_names)
        if retained_names:
            raise ValueError(
                "Relocated symbols remain referenced by their source module: "
                f"{tuple(sorted(retained_names))!r}"
            )
        return ()


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
            owner = cls._declaring_path(context, binding_module, name)
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

    @staticmethod
    def _declaring_path(
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
class _ModuleIdentityStrings:
    """String constants naming the dissolved module: its dotted name, its path,
    or `<module>.<Name>` (a patch target); each is rewritten to its owner."""

    source_path: str
    source_module_name: str
    owners: Mapping[str, str]
    successor_path: str | None

    def edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        edits: list[NominalSourceEdit] = []
        for source_file in context.source_index.files:
            file_path = source_file.file_path
            module = context.module_nodes_by_file_path.get(file_path)
            if file_path == self.source_path or module is None:
                continue
            replacements = tuple(
                (node, replacement)
                for node in ast.walk(module)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                for replacement in (self._replacement(context, file_path, node),)
                if replacement is not None
            )
            if not replacements:
                continue
            geometry = SourceTextGeometry(context.sources_by_file_path[file_path])
            edits.append(
                geometry.nominal_edit(
                    file_path=file_path,
                    replacements=tuple(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=end,
                            replacement_source=repr(replacement),
                        )
                        for node, replacement in replacements
                        for start, end in (geometry.required_node_offsets(node),)
                    ),
                )
            )
        return tuple(edits)

    def _replacement(
        self,
        context: CodemodSelectorContext,
        file_path: str,
        node: ast.Constant,
    ) -> str | None:
        value = node.value
        import_graph = context.module_import_graph
        prefix = f"{self.source_module_name}."
        if value.startswith(prefix) and value[len(prefix):] in self.owners:
            name = value[len(prefix):]
            return f"{import_graph.module_name_for_file_path(self.owners[name])}.{name}"
        names_module = value == self.source_module_name or value.startswith(prefix)
        names_path = value.endswith(".py") and (
            value == self.source_path or self.source_path.endswith(f"/{value}")
        )
        if not (names_module or names_path):
            return None
        if self.successor_path is None or value.startswith(prefix):
            raise ValueError(
                f"{file_path!r}:{node.lineno} names the dissolved module as {value!r}; "
                "declare its successor_path"
            )
        if names_module:
            return import_graph.module_name_for_file_path(self.successor_path)
        root = self.source_path[: len(self.source_path) - len(value)]
        if not self.successor_path.startswith(root):
            raise ValueError(
                f"{file_path!r}:{node.lineno} names {value!r} relative to a root "
                f"{self.successor_path!r} does not share"
            )
        return self.successor_path[len(root):]


@dataclass(frozen=True)
class DissolvedModuleReport(DataclassJsonReport):
    """Where a dissolved module's names went."""

    deleted_path: str
    owner_paths_by_name: tuple[tuple[str, str], ...]


@dataclass(frozen=True, kw_only=True)
class DissolveModuleOperation(RepositorySourceReprovedOperation):
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
        self._require_followable_consumers(context, source_path, source_module_name, bindings)
        return (
            *ModuleNameRedirect(source_path, bindings.owners).consumer_edits(
                context,
                excluded_paths=excluded,
            ),
            *self._module_binding_removals(context, source_path, source_module_name),
            *_ModuleIdentityStrings(
                source_path,
                source_module_name,
                bindings.owners,
                successor_path,
            ).edits(context),
        )

    @staticmethod
    def _require_followable_consumers(
        context: CodemodSelectorContext,
        source_path: str,
        source_module_name: str,
        bindings: _DissolvedModuleBindings,
    ) -> None:
        for source_file in context.source_index.files:
            module = context.module_nodes_by_file_path.get(source_file.file_path)
            if source_file.file_path == source_path or module is None:
                continue
            identity = source_file.module_path_identity
            for node in ast.walk(module):
                if not isinstance(node, ast.ImportFrom) or (
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
    def _module_binding_removals(
        context: CodemodSelectorContext,
        source_path: str,
        source_module_name: str,
    ) -> Iterator[NominalSourceEdit]:
        """Remove `from pkg import source` bindings once their reads are redirected.

        `ModuleNameRedirect` refuses any use other than an attribute read, so a
        binding still standing here is either all reads (now redirected) or unused,
        which means it was taken for its import side effect."""

        package, _, leaf = source_module_name.rpartition(".")
        for source_file in context.source_index.files:
            module = context.module_nodes_by_file_path.get(source_file.file_path)
            if source_file.file_path == source_path or module is None:
                continue
            identity = source_file.module_path_identity
            scoped = {
                id(statement): scope
                for scope in ModuleImportScope
                for statement in scope.import_statements(module)
            }
            for scope, statement in (
                (scoped.get(id(node)), node)
                for node in ast.walk(module)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ):
                if isinstance(statement, ast.Import):
                    if any(
                        alias.name == source_module_name
                        or alias.name.startswith(f"{source_module_name}.")
                        for alias in statement.names
                    ):
                        raise ValueError(
                            f"{source_file.file_path!r}:{statement.lineno} imports "
                            f"{source_module_name!r} as a module path"
                        )
                    continue
                if (
                    identity.resolve_import_from_module(
                        imported_module=statement.module,
                        level=statement.level,
                    )
                    != package
                ):
                    continue
                aliases = tuple(alias for alias in statement.names if alias.name == leaf)
                for alias in aliases:
                    bound = alias.asname or alias.name
                    if not any(
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == bound
                        for node in ast.walk(module)
                    ):
                        raise ValueError(
                            f"{source_file.file_path!r}:{statement.lineno} binds "
                            f"{source_module_name!r} for its import side effect"
                        )
                if aliases and scope is None:
                    raise ValueError(
                        f"{source_file.file_path!r}:{statement.lineno} binds "
                        f"{source_module_name!r} inside a function"
                    )
                if aliases:
                    yield ModuleImportMutation.remove_names(
                        file_path=source_file.file_path,
                        module_name=ImportFromModuleName.from_node(statement).source,
                        names=(alias.name for alias in aliases),
                        scope=scope,
                    )
