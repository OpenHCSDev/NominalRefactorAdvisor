"""Shared lexical binding and import-origin declarations."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Self

from .assignment_projection import AssignmentTargetNameProjection
from .python_module_identity import PythonModulePathIdentity


class CompactParameterKind(StrEnum):
    """Python parameter kinds with their binding behavior on each member."""

    POSITIONAL_ONLY = "positional_only", True, False, False
    POSITIONAL_OR_KEYWORD = "positional_or_keyword", True, True, False
    VAR_POSITIONAL = "var_positional", True, False, True
    KEYWORD_ONLY = "keyword_only", False, True, False
    VAR_KEYWORD = "var_keyword", False, True, True

    def __new__(
        cls,
        value: str,
        accepts_positional: bool,
        accepts_keyword: bool,
        variadic: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._accepts_positional = accepts_positional
        member._accepts_keyword = accepts_keyword
        member._variadic = variadic
        return member

    @property
    def accepts_positional(self) -> bool:
        return self._accepts_positional

    @property
    def accepts_keyword(self) -> bool:
        return self._accepts_keyword

    @property
    def variadic(self) -> bool:
        return self._variadic


@dataclass(frozen=True)
class FunctionParameterSource:
    """Actual argument/default nodes in signature order, not execution evidence."""

    argument: ast.arg
    kind: CompactParameterKind
    default: ast.expr | None = None

    @classmethod
    def from_arguments(cls, arguments: ast.arguments) -> tuple[Self, ...]:
        positional = (*arguments.posonlyargs, *arguments.args)
        defaults = (None,) * (len(positional) - len(arguments.defaults)) + tuple(
            arguments.defaults
        )
        parameters = [
            cls(
                argument,
                (
                    CompactParameterKind.POSITIONAL_ONLY
                    if index < len(arguments.posonlyargs)
                    else CompactParameterKind.POSITIONAL_OR_KEYWORD
                ),
                default,
            )
            for index, (argument, default) in enumerate(
                zip(positional, defaults, strict=True)
            )
        ]
        if arguments.vararg is not None:
            parameters.append(
                cls(arguments.vararg, CompactParameterKind.VAR_POSITIONAL)
            )
        parameters.extend(
            cls(argument, CompactParameterKind.KEYWORD_ONLY, default)
            for argument, default in zip(
                arguments.kwonlyargs, arguments.kw_defaults, strict=True
            )
        )
        if arguments.kwarg is not None:
            parameters.append(cls(arguments.kwarg, CompactParameterKind.VAR_KEYWORD))
        return tuple(parameters)


class FunctionDefaultVisitor(ast.NodeVisitor):
    """Visit defaults when a function is created, without entering its lambda body."""

    def visit_argument_defaults(self, arguments: ast.arguments) -> None:
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_argument_defaults(node.args)


@dataclass(frozen=True)
class ImportAliasRequirement:
    """One requested import alias, including alias spelling when present."""

    name: str
    asname: str | None

    @property
    def source(self) -> str:
        return self.name if self.asname is None else f"{self.name} as {self.asname}"

    @classmethod
    def from_alias(cls, alias: ast.alias) -> "ImportAliasRequirement":
        return cls(name=alias.name, asname=alias.asname)

    @property
    def canonical_key(self) -> tuple[str, str]:
        """Return the source-spelling key for commutative import merging."""

        return self.name, self.asname or ""


@dataclass(frozen=True)
class ImportFromModuleName:
    """Canonical source spelling for an ImportFrom module."""

    source: str

    def resolve(self, module_identity: PythonModulePathIdentity) -> str | None:
        module = self.source.lstrip(".")
        return module_identity.resolve_import_from_module(
            imported_module=module or None,
            level=len(self.source) - len(module),
        )

    @classmethod
    def from_node(cls, node: ast.ImportFrom) -> "ImportFromModuleName":
        relative_prefix = "." * node.level
        if node.module is None:
            return cls(relative_prefix)
        return cls(f"{relative_prefix}{node.module}")


@dataclass(frozen=True)
class ImportDeclarationABC(ABC):
    """One AST-free import request; binding views derive from its aliases."""

    aliases: tuple[ImportAliasRequirement, ...]

    @property
    @abstractmethod
    def source_prefix(self) -> str:
        raise NotImplementedError

    def source_for(self, aliases: tuple[ImportAliasRequirement, ...]) -> str:
        return f"{self.source_prefix} {', '.join(alias.source for alias in aliases)}\n"

    @property
    def source(self) -> str:
        return self.source_for(self.aliases)

    def bound_name(self, alias: ImportAliasRequirement) -> str | None:
        if alias.name == "*":
            return None
        return alias.asname or self._unaliased_bound_name(alias)

    @abstractmethod
    def _unaliased_bound_name(self, alias: ImportAliasRequirement) -> str:
        raise NotImplementedError

    @abstractmethod
    def requested_module_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def qualified_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        """A source catalogue path, not a claim about the imported runtime object."""
        raise NotImplementedError

    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for alias in self.aliases
            if (name := self.bound_name(alias)) is not None
        )

    def origins(
        self, module_identity: PythonModulePathIdentity
    ) -> tuple[ImportedNameOrigin, ...]:
        return tuple(
            ImportedNameOrigin(self, index, module_identity)
            for index, alias in enumerate(self.aliases)
            if self.bound_name(alias) is not None
        )

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.source_for((alias,)))
            for alias in self.aliases
            if (name := self.bound_name(alias)) is not None
        )


class ModuleImportDeclaration(ImportDeclarationABC):
    source_prefix = "import"

    def _unaliased_bound_name(self, alias: ImportAliasRequirement) -> str:
        return alias.name.partition(".")[0]

    def requested_module_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return alias.name

    def qualified_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return (
            alias.name
            if alias.asname is not None
            else self._unaliased_bound_name(alias)
        )


@dataclass(frozen=True)
class FromImportDeclaration(ImportDeclarationABC):
    module_name: ImportFromModuleName

    @property
    def source_prefix(self) -> str:
        return f"from {self.module_name.source} import"

    def _unaliased_bound_name(self, alias: ImportAliasRequirement) -> str:
        return alias.name

    def requested_module_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        return self.module_name.resolve(module_identity)

    def qualified_name(
        self,
        alias: ImportAliasRequirement,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        module_name = self.requested_module_name(alias, module_identity)
        return None if module_name is None else f"{module_name}.{alias.name}"


@dataclass(frozen=True)
class ImportedNameOrigin:
    """A selected source position in the actual import declaration."""

    declaration: ImportDeclarationABC
    alias_index: int
    module_identity: PythonModulePathIdentity

    def __post_init__(self) -> None:
        if not 0 <= self.alias_index < len(self.declaration.aliases):
            raise ValueError(
                "Import origin requires an alias position within its declaration"
            )
        if self.declaration.bound_name(self.alias) is None:
            raise ValueError("A star request has no explicit individual binding")

    @property
    def alias(self) -> ImportAliasRequirement:
        return self.declaration.aliases[self.alias_index]

    @property
    def bound_name(self) -> str:
        name = self.declaration.bound_name(self.alias)
        assert name is not None
        return name

    @property
    def qualified_name(self) -> str | None:
        return self.declaration.qualified_name(self.alias, self.module_identity)

    @property
    def requested_module_name(self) -> str | None:
        return self.declaration.requested_module_name(self.alias, self.module_identity)

    @property
    def source(self) -> str:
        return self.declaration.source_for((self.alias,))


@dataclass(frozen=True)
class ImportBoundNameProjection:
    """AST bridge to the declaration owning import syntax and binding rules."""

    statement: ast.Import | ast.ImportFrom

    @cached_property
    def declaration(self) -> ImportDeclarationABC:
        aliases = tuple(
            ImportAliasRequirement.from_alias(alias) for alias in self.statement.names
        )
        if isinstance(self.statement, ast.Import):
            return ModuleImportDeclaration(aliases)
        return FromImportDeclaration(
            aliases, ImportFromModuleName.from_node(self.statement)
        )

    def names(self) -> tuple[str, ...]:
        return self.declaration.names()

    def origins(
        self, module_identity: PythonModulePathIdentity
    ) -> tuple[ImportedNameOrigin, ...]:
        return self.declaration.origins(module_identity)

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return self.declaration.name_sources()

    def alias_bound_name(self, alias: ast.alias) -> str:
        return (
            self.declaration.bound_name(ImportAliasRequirement.from_alias(alias)) or ""
        )


@dataclass
class ScopeBindingCollector(ast.NodeVisitor):
    """Collect compile-time bindings without descending into child scopes."""

    bound_names: set[str] = field(default_factory=set)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.bound_names.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)

    def visit_Import(self, node: ast.Import) -> None:
        self.bound_names.update(ImportBoundNameProjection(node).names())

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bound_names.update(ImportBoundNameProjection(node).names())

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.bound_names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.bound_names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.bound_names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.bound_names.add(node.rest)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        _ComprehensionContainingScopeBindingCollector(self.bound_names).generic_visit(
            node
        )

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class _ComprehensionContainingScopeBindingCollector(ScopeBindingCollector):
    """Only walrus targets escape a comprehension; headers retain normal ownership."""

    def visit_Name(self, node: ast.Name) -> None:
        return

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.bound_names.update(_store_names(node.target))
        self.visit(node.value)


def _store_names(target: ast.expr) -> tuple[str, ...]:
    return AssignmentTargetNameProjection(target).names


class LexicalScopeBindingAuthority:
    """Recover names bound by one lexical scope without entering child scopes."""

    @staticmethod
    def bound_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
        collector = ScopeBindingCollector()
        for node in nodes:
            collector.visit(node)
        return frozenset(collector.bound_names)

    @staticmethod
    def argument_names(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> frozenset[str]:
        arguments = node.args
        return frozenset(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ) | frozenset(
            argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument
        )


LEXICAL_SCOPE_BINDING_AUTHORITY = LexicalScopeBindingAuthority()
