"""Canonical binding identities for Python import refactors."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .codemod_import_graph import SourceModuleImportGraph
from .codemod_import_scopes import ModuleImportScope
from .codemod_imports import ImportFromSource, RequestedImportStatement
from .json_reports import (
    DataclassJsonReport,
    json_report_property,
)


@dataclass(frozen=True)
class ModuleImportBinding:
    """One single-alias import declaration and its bound name."""

    name: str
    request: RequestedImportStatement

    @property
    def source(self) -> str:
        return self.request.source

    @property
    def scope(self) -> ModuleImportScope:
        return self.request.scope

    @property
    def supports_bound_name_removal(self) -> bool:
        """Return whether this declaration explicitly owns the bound alias."""

        return True

    def identity(
        self,
        import_graph: SourceModuleImportGraph,
        importing_file_path: str,
    ) -> "ModuleImportBindingIdentity | None":
        statement = self.request.statement
        alias = statement.names[0]
        if isinstance(statement, ast.Import):
            return DirectModuleImportBindingIdentity(
                module_name=alias.name,
            )
        importing_file = import_graph.source_file_for_path(importing_file_path)
        if importing_file is None:
            return None
        module_name = import_graph.resolve_import_from_module(
            importing_file,
            imported_module=statement.module,
            level=statement.level,
        )
        if module_name is None:
            return None
        return FromModuleImportBindingIdentity(
            module_name=module_name,
            member_name=alias.name,
        )


@dataclass(frozen=True)
class ModuleImportBindingIdentity(DataclassJsonReport, ABC):
    """Canonical import identity independent of source spelling."""

    module_name: str

    @json_report_property()
    @abstractmethod
    def imported_name(self) -> str | None:
        """Return the from-import member, when this identity has one."""

        raise NotImplementedError

    def is_destination_declaration(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        destination_path: str,
        bound_name: str,
    ) -> bool:
        """Return whether this import names a declaration owned by destination."""

        return False

    @staticmethod
    def alias_for_bound_name(
        source_name: str,
        default_bound_name: str,
        bound_name: str,
    ) -> ast.alias:
        """Derive alias syntax from imported identity and required binding."""

        return ast.alias(
            name=source_name,
            asname=None if bound_name == default_bound_name else bound_name,
        )

    @abstractmethod
    def source_for(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        importing_file_path: str,
        scope: ModuleImportScope,
        bound_name: str,
    ) -> str:
        """Render this binding relative to its new importing module."""

        raise NotImplementedError


@dataclass(frozen=True)
class DirectModuleImportBindingIdentity(ModuleImportBindingIdentity):
    """Canonical identity and rendering for ``import module`` bindings."""

    @property
    def imported_name(self) -> None:
        return None

    def source_for(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        importing_file_path: str,
        scope: ModuleImportScope,
        bound_name: str,
    ) -> str:
        del import_graph, importing_file_path
        alias = self.alias_for_bound_name(
            self.module_name,
            self.module_name.partition(".")[0],
            bound_name,
        )
        return RequestedImportStatement(
            ast.Import(names=[alias]),
            scope=scope,
        ).source


@dataclass(frozen=True)
class FromModuleImportBindingIdentity(ModuleImportBindingIdentity):
    """Canonical identity and destination rendering for from-import bindings."""

    member_name: str

    @property
    def imported_name(self) -> str:
        return self.member_name

    def is_destination_declaration(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        destination_path: str,
        bound_name: str,
    ) -> bool:
        return (
            self.imported_name == bound_name
            and self.module_name
            == import_graph.module_name_for_file_path(destination_path)
        )

    def source_for(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        importing_file_path: str,
        scope: ModuleImportScope,
        bound_name: str,
    ) -> str:
        imported_file = import_graph.unique_source_file_for_module_name(
            self.module_name
        )
        module_reference = self.module_name
        if imported_file is not None:
            module_reference = scope.required_module_reference(
                import_graph,
                importing_file_path=importing_file_path,
                imported_file_path=imported_file.file_path,
                imported_name=self.imported_name,
            )
        return ImportFromSource(
            module_name=module_reference,
            aliases=(
                self.alias_for_bound_name(
                    self.imported_name,
                    self.imported_name,
                    bound_name,
                ),
            ),
        ).source
