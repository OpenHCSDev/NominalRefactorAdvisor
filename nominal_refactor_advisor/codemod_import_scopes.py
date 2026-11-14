"""Declaration-owned execution scopes for Python import mutations."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias
from nominal_refactor_advisor.fielded_enum import FieldedStrEnum
from nominal_refactor_advisor.lexical_bindings import ImportBoundNameProjection


if TYPE_CHECKING:
    from nominal_refactor_advisor.codemod_import_graph import SourceModuleImportGraph


ImportStatement: TypeAlias = ast.Import | ast.ImportFrom


def _runtime_import_statements(module: ast.Module) -> tuple[ImportStatement, ...]:
    return tuple(
        statement
        for statement in module.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    )


def _type_checking_import_statements(
    module: ast.Module,
) -> tuple[ImportStatement, ...]:
    return tuple(
        statement
        for guard in TypeCheckingGuardProjection.from_module(module).guards
        for statement in guard.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    )


def _runtime_module_reference(
    import_graph: SourceModuleImportGraph,
    importing_file_path: str,
    imported_file_path: str,
    imported_name: str,
) -> str:
    return import_graph.required_import_module_reference(
        importing_file_path=importing_file_path,
        imported_file_path=imported_file_path,
        imported_name=imported_name,
    )


def _guarded_module_reference(
    import_graph: SourceModuleImportGraph,
    importing_file_path: str,
    imported_file_path: str,
    imported_name: str,
) -> str:
    module_reference = import_graph.import_module_reference(
        importing_file_path=importing_file_path,
        imported_file_path=imported_file_path,
        imported_name=imported_name,
    )
    if module_reference is None:
        raise ValueError(
            f"No canonical import exists for {imported_name!r} from "
            f"{imported_file_path!r} into {importing_file_path!r}"
        )
    return module_reference


class ModuleImportScope(FieldedStrEnum):
    """Execution scope carried by every import requirement and mutation."""

    RUNTIME = (
        "runtime",
        0,
        False,
        _runtime_import_statements,
        _runtime_module_reference,
    )
    TYPE_CHECKING = (
        "type_checking",
        1,
        True,
        _type_checking_import_statements,
        _guarded_module_reference,
    )

    canonical_rank: int
    """Return source order without creating a second priority registry."""
    is_guarded: bool
    """Return whether imports execute only for static type checking."""
    _statement_projection: Callable[[ast.Module], tuple[ImportStatement, ...]]
    _module_reference_resolver: Callable[[SourceModuleImportGraph, str, str, str], str]

    def import_statements(self, module: ast.Module) -> tuple[ImportStatement, ...]:
        """Project imports belonging to this declaration-owned scope."""

        return self._statement_projection(module)

    def bindings(self, module: ast.Module) -> Iterator[str]:
        """The name each alias of this scope's imports binds, once per alias."""

        for statement in self.import_statements(module):
            yield from ImportBoundNameProjection(statement).names()

    @staticmethod
    def bound_in(
        module: ast.Module, names: Iterable[str]
    ) -> Iterator[tuple[ModuleImportScope, tuple[str, ...]]]:
        """Each scope whose imports bind some of `names`, with those names."""

        requested = tuple(dict.fromkeys(names))
        for scope in ModuleImportScope:
            bound = frozenset(scope.bindings(module))
            matched = tuple(name for name in requested if name in bound)
            if matched:
                yield scope, matched

    def required_module_reference(
        self,
        import_graph: SourceModuleImportGraph,
        *,
        importing_file_path: str,
        imported_file_path: str,
        imported_name: str,
    ) -> str:
        """Resolve an import path with the cycle policy carried by this scope."""

        return self._module_reference_resolver(
            import_graph,
            importing_file_path,
            imported_file_path,
            imported_name,
        )

    def is_satisfied_by(self, available_scope: ModuleImportScope) -> bool:
        """Return whether an existing import is visible wherever this scope is."""

        return self.is_guarded or not available_scope.is_guarded


@dataclass(frozen=True)
class TypeCheckingGuardReference:
    """One syntax reference proved to denote ``typing.TYPE_CHECKING``."""

    expression: ast.expr

    @property
    def source(self) -> str:
        return ast.unparse(self.expression)

    def matches(self, expression: ast.expr) -> bool:
        return ast.dump(self.expression, include_attributes=False) == ast.dump(
            expression,
            include_attributes=False,
        )


@dataclass(frozen=True)
class TypeCheckingGuardProjection:
    """Guards derived from runtime imports of ``typing.TYPE_CHECKING``."""

    references: tuple[TypeCheckingGuardReference, ...]
    guards: tuple[ast.If, ...]

    @classmethod
    def from_module(cls, module: ast.Module) -> TypeCheckingGuardProjection:
        references = cls._references(module)
        return cls(
            references=references,
            guards=tuple(
                statement
                for statement in module.body
                if isinstance(statement, ast.If)
                and any(reference.matches(statement.test) for reference in references)
            ),
        )

    @classmethod
    def _references(
        cls,
        module: ast.Module,
    ) -> tuple[TypeCheckingGuardReference, ...]:
        return tuple(
            reference
            for statement in module.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            for reference in cls._references_for_statement(statement)
        )

    @staticmethod
    def _references_for_statement(
        statement: ImportStatement,
    ) -> tuple[TypeCheckingGuardReference, ...]:
        if (
            isinstance(statement, ast.ImportFrom)
            and statement.level == 0
            and statement.module == "typing"
        ):
            return tuple(
                TypeCheckingGuardReference(
                    ast.Name(id=ImportBoundNameProjection(statement).alias_bound_name(alias), ctx=ast.Load())
                )
                for alias in statement.names
                if alias.name == "TYPE_CHECKING"
            )
        if isinstance(statement, ast.Import):
            return tuple(
                TypeCheckingGuardReference(
                    ast.Attribute(
                        value=ast.Name(
                            id=ImportBoundNameProjection(statement).alias_bound_name(alias),
                            ctx=ast.Load(),
                        ),
                        attr="TYPE_CHECKING",
                        ctx=ast.Load(),
                    )
                )
                for alias in statement.names
                if alias.name == "typing"
            )
        return ()

    @property
    def preferred_reference_source(self) -> str | None:
        if not self.references:
            return None
        return self.references[0].source
