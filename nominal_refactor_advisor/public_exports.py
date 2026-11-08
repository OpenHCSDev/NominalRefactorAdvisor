"""A module's public export contract: its __all__ declaration or Python's implicit underscore convention."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import (
    Self,
    cast,
)

from .ast_projection import (
    StatementValues,
    single_name_targets,
)
from .ast_tools import ParsedModule
from .collection_algebra import sorted_tuple
from .export_tools import PYTHON_PUBLIC_EXPORT_ASSIGNMENT
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY


class CompactPublicNameExposure(StrEnum):
    """Proof result for one name on a module's declared public surface."""

    PUBLIC = "public", True
    PRIVATE = "private", False
    UNRESOLVED = "unresolved", True

    def __new__(cls, value: str, blocks_closed_boundary: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._blocks_closed_boundary = blocks_closed_boundary
        return member

    @property
    def blocks_closed_boundary(self) -> bool:
        return self._blocks_closed_boundary

    @property
    def proves_public_exposure(self) -> bool:
        return self is type(self).PUBLIC

    @property
    def introduces_uncertainty(self) -> bool:
        return self is type(self).UNRESOLVED


class CompactModulePublicExportContract(ABC):
    """Representation-independent declaration of one module's export policy."""

    @abstractmethod
    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def allows_binding_relocation(self, name: str) -> bool:
        """Return whether removing this module binding preserves the contract."""

        raise NotImplementedError


@dataclass(frozen=True)
class CompactImplicitPublicExportContract(CompactModulePublicExportContract):
    """Python's implicit convention: leading-underscore bindings are private."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PRIVATE
            if name.startswith("_")
            else CompactPublicNameExposure.PUBLIC
        )

    def allows_binding_relocation(self, name: str) -> bool:
        del name
        return True


@dataclass(frozen=True)
class CompactExplicitPublicExportContract(CompactModulePublicExportContract):
    """One statically complete ``__all__`` membership declaration."""

    exported_names: tuple[str, ...]

    @classmethod
    def from_declaration(
        cls,
        declaration: ModulePublicExportSourceAuthority,
    ) -> CompactExplicitPublicExportContract | None:
        references = declaration.literal_references
        if references is None:
            return None
        return cls(sorted_tuple({reference.name for reference in references}))

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PUBLIC
            if name in self.exported_names
            else CompactPublicNameExposure.PRIVATE
        )

    def allows_binding_relocation(self, name: str) -> bool:
        return name not in self.exported_names


@dataclass(frozen=True)
class CompactUnresolvedPublicExportContract(CompactModulePublicExportContract):
    """A dynamic or otherwise incomplete ``__all__`` declaration."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        del name
        return CompactPublicNameExposure.UNRESOLVED

    def allows_binding_relocation(self, name: str) -> bool:
        del name
        return False


class _ModuleScopeNameReferenceCollector(ast.NodeVisitor):
    """Collect one module binding's syntax without entering child namespaces."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.references: list[ast.Name] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name:
            self.references.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _module_scope_name_references(
    module: ast.Module,
    name: str,
) -> tuple[ast.Name, ...]:
    collector = _ModuleScopeNameReferenceCollector(name)
    collector.visit(module)
    return tuple(collector.references)


@dataclass(frozen=True)
class PublicExportNameReference:
    """One exact string literal in a static public-export declaration."""

    literal: ast.Constant

    @property
    def name(self) -> str:
        return cast(str, self.literal.value)

    def __post_init__(self) -> None:
        if not isinstance(self.literal.value, str):
            raise ValueError("Public export reference must contain a string")

    def renamed_source(self, literal_source: str, new_name: str) -> str:
        if literal_source.count(self.name) != 1:
            raise ValueError("Public export reference cannot be reconstructed")
        return literal_source.replace(self.name, new_name, 1)


@dataclass(frozen=True)
class ModulePublicExportSourceAuthority:
    """Exact source declaration from which a module export contract is derived."""

    statement: ast.Assign | ast.AnnAssign
    target: ast.Name
    value: ast.expr

    @cached_property
    def literal_references(self) -> tuple[PublicExportNameReference, ...] | None:
        """Validate the whole literal sequence before exposing any member sites."""
        if not isinstance(self.value, ast.List | ast.Tuple):
            return None
        references = []
        for element in self.value.elts:
            if not isinstance(element, ast.Constant):
                return None
            try:
                references.append(PublicExportNameReference(element))
            except ValueError:
                return None
        return tuple(references)

    @staticmethod
    def export_targets(
        statement: ast.Assign | ast.AnnAssign,
    ) -> tuple[ast.Name, ...]:
        """The `__all__` name a module-scope assignment binds, if it binds it."""

        return tuple(
            target
            for target in single_name_targets(statement)
            if target.id == PYTHON_PUBLIC_EXPORT_ASSIGNMENT
        )

    @classmethod
    def from_module(
        cls,
        module: ast.Module,
    ) -> ModulePublicExportSourceAuthority | None:
        # A bare `__all__: T` declares no export list.
        declarations = tuple(
            cls(statement, target, supplied.value)
            for (target, statement), supplied in StatementValues.keyed(
                ((target, statement), statement)
                for statement in module.body
                if isinstance(statement, ast.Assign | ast.AnnAssign)
                for target in cls.export_targets(statement)
            ).supplied
        )
        if len(declarations) != 1:
            return None
        declaration = declarations[0]
        references = _module_scope_name_references(
            module,
            PYTHON_PUBLIC_EXPORT_ASSIGNMENT,
        )
        return declaration if references == (declaration.target,) else None

    def name_references(self, name: str) -> tuple[PublicExportNameReference, ...]:
        references = self.literal_references
        if references is None:
            return ()
        return tuple(reference for reference in references if reference.name == name)


def module_public_export_contract(
    parsed_module: ParsedModule,
) -> CompactModulePublicExportContract:
    module = parsed_module.module
    if PYTHON_PUBLIC_EXPORT_ASSIGNMENT not in (
        LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
    ):
        return CompactImplicitPublicExportContract()
    declaration = ModulePublicExportSourceAuthority.from_module(module)
    if declaration is None:
        return CompactUnresolvedPublicExportContract()
    return (
        CompactExplicitPublicExportContract.from_declaration(declaration)
        or CompactUnresolvedPublicExportContract()
    )
