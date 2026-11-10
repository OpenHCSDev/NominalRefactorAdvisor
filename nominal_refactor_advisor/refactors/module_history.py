"""Replay recorded provider-module moves onto callers without loading either app."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path, PurePosixPath

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleSet, SourceModule
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.audit.repository import Repository
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.python_module_identity import PythonModulePathIdentity
from nominal_refactor_advisor.refactors.module_move import FilesReachingModule, WholeModuleNameRedirect
from nominal_refactor_advisor.source_identity import PACKAGE_INIT_FILE_NAME


@dataclass(frozen=True)
class _GitPythonRevision:
    """Python package identities and source at one immutable Git revision."""

    repository: Repository
    revision: str

    @cached_property
    def paths(self) -> frozenset[str]:
        return self.repository.python_paths(self.revision, ".")

    @cache
    def identity(self, relative: str) -> PythonModulePathIdentity:
        directory = PurePosixPath(relative).parent
        while str(directory / PACKAGE_INIT_FILE_NAME) in self.paths:
            directory = directory.parent
        root = self.repository.path / ".nra-git-source" / self.revision
        return PythonModulePathIdentity.from_import_root(root / relative, root / directory)

    def module(self, relative: str) -> ParsedModule:
        return SourceModule.from_path_identity(
            self.identity(relative), self.repository.source(self.revision, relative)
        ).parse()


@dataclass(frozen=True)
class _MovedProviderBindings(Mapping[str, str]):
    """An actual required binding must still exist at its recorded destination."""

    source_names: frozenset[str]
    destination_names: frozenset[str]
    destination: str

    def __iter__(self) -> Iterator[str]:
        return iter(self.source_names)

    def __len__(self) -> int:
        return len(self.source_names)

    def __getitem__(self, name: str) -> str:
        if name not in self.source_names:
            raise KeyError(name)
        if name not in self.destination_names:
            raise ValueError(f"Recorded move loses provider bindings: {name}")
        return self.destination


@dataclass(frozen=True)
class _HistoryModuleRedirect(WholeModuleNameRedirect):
    """An existing caller keeps its import at the same execution boundary."""

    @staticmethod
    def _sorted_block_import_ids(source: str, file_path: str, module: ast.Module) -> frozenset[int]:
        return frozenset()

    def consumer_edits(
        self, context: CodemodSelectorContext, *, excluded_paths: frozenset[str]
    ) -> tuple[NominalSourceEdit, ...]:
        source_name = self.source_module_name(context.module_import_graph)
        for source, parsed in FilesReachingModule(context, source_name):
            if source.file_path in excluded_paths:
                continue
            for statement in parsed.fact(ModuleNodeIndex).import_froms():
                if source.module_path_identity.resolve_import_from_module(
                    imported_module=statement.module, level=statement.level
                ) == source_name and any(alias.name == "*" for alias in statement.names):
                    raise ValueError("Historical star imports need an explicit import-order proof")
        return super().consumer_edits(context, excluded_paths=excluded_paths)


@dataclass(frozen=True)
class _RecordedModuleMove:
    before: ParsedModule
    after: ParsedModule

    @staticmethod
    def bindings(module: ParsedModule) -> frozenset[str]:
        table = ModuleSymbolTable(
            file_path=module.file_path, source=module.source, module=module.module
        )
        return table.explicit_names

    @cached_property
    def redirect(self) -> _HistoryModuleRedirect:
        names = self.bindings(self.before)
        return _HistoryModuleRedirect(
            source_path=self.before.file_path,
            destinations=_MovedProviderBindings(
                names, self.bindings(self.after), self.after.file_path
            ),
            reader_path=self.after.file_path,
            reader_bindings=names,
            home=self.after.module_name,
        )


    def consumer_edits(
        self, context: CodemodSourceSnapshot, providers: frozenset[str]
    ) -> tuple[NominalSourceEdit, ...]:
        return self.redirect.consumer_edits(context, excluded_paths=providers)


@dataclass(frozen=True, kw_only=True)
class ReplayGitModuleImportsOperation(RepositorySourceReprovedOperation):
    """Repair caller imports from Git-recorded moves; provider sources are read-only.

    References resolve to immutable commits at construction, so applying a saved
    plan cannot silently follow a branch that changed after planning. Package
    identities come from the recorded package files, not the current checkout.
    """

    provider_repository: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_revision: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    target_revision: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    caller_root: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        repository = Repository(Path(self.provider_repository).resolve())
        caller = Path(self.caller_root).resolve()
        if not caller.is_dir():
            raise ValueError(f"Caller root is not a directory: {caller}")
        if caller.is_relative_to(repository.path) or repository.path.is_relative_to(caller):
            raise ValueError("Provider repository and caller tree must be separate")
        object.__setattr__(self, "provider_repository", str(repository.path))
        object.__setattr__(self, "caller_root", str(caller))
        object.__setattr__(self, "source_revision", repository.revision(self.source_revision))
        object.__setattr__(self, "target_revision", repository.revision(self.target_revision))

    def recorded_moves(self, callers: tuple[ParsedModule, ...]) -> tuple[_RecordedModuleMove, ...]:
        repository = Repository(Path(self.provider_repository))
        before = _GitPythonRevision(repository, self.source_revision)
        after = _GitPythonRevision(repository, self.target_revision)
        caller_sources = tuple(module.source for module in callers)
        moves = tuple(
            _RecordedModuleMove(before.module(old), after.module(new))
            for change in repository.changes(self.source_revision, self.target_revision)
            for old, new in change.moved()
            if repository.python(old) and repository.python(new)
            and before.identity(old).import_name != after.identity(new).import_name
            and any(
                before.identity(old).import_name.rpartition(".")[2] in source
                for source in caller_sources
            )
        )
        old_names = frozenset(move.before.module_name for move in moves)
        new_names = frozenset(move.after.module_name for move in moves)
        if old_names & new_names:
            raise ValueError("Recorded module moves overlap old and new namespaces")
        if len(old_names) != len(moves) or len(new_names) != len(moves):
            raise ValueError("Recorded module moves have ambiguous package identities")
        return moves

    def source_edits_from_snapshot(
        self, context: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        caller_root = Path(self.caller_root)
        callers = tuple(
            module for module in context.parsed_modules
            if Path(module.file_path).is_relative_to(caller_root)
        )
        moves = self.recorded_moves(callers)
        if not moves:
            return ()
        provider_modules = tuple(module for move in moves for module in (move.before, move.after))
        proof = CodemodSourceSnapshot.from_modules(ParsedModuleSet((*callers, *provider_modules)))
        providers = frozenset(module.file_path for module in provider_modules)
        return tuple(
            edit
            for move in moves
            for edit in move.consumer_edits(proof, providers)
        )
