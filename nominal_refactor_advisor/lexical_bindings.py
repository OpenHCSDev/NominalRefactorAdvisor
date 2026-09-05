"""Shared lexical binding and import-origin declarations."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field

from .assignment_projection import AssignmentTargetNameProjection
from .python_module_identity import PythonModulePathIdentity


@dataclass(frozen=True)
class ImportedNameOrigin:
    """One explicitly bound import name and its resolved nominal origin."""

    bound_name: str
    qualified_name: str | None


@dataclass(frozen=True)
class ImportBoundNameProjection:
    """Project Python import statements to names bound in their lexical scope."""

    statement: ast.Import | ast.ImportFrom

    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for alias in self.statement.names
            if (name := self.alias_bound_name(alias))
        )

    def origins(
        self, module_identity: PythonModulePathIdentity
    ) -> tuple[ImportedNameOrigin, ...]:
        return tuple(
            ImportedNameOrigin(bound_name, self.alias_origin(alias, module_identity))
            for alias in self.statement.names
            if (bound_name := self.alias_bound_name(alias))
        )

    def alias_origin(
        self, alias: ast.alias, module_identity: PythonModulePathIdentity
    ) -> str | None:
        if isinstance(self.statement, ast.Import):
            return alias.name if alias.asname else alias.name.split(".", 1)[0]
        module_name = module_identity.resolve_import_from_module(
            imported_module=self.statement.module,
            level=self.statement.level,
        )
        return None if module_name is None else f"{module_name}.{alias.name}"

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.alias_import_source(alias))
            for alias in self.statement.names
            for name in (self.alias_bound_name(alias),)
            if name
        )

    def alias_bound_name(self, alias: ast.alias) -> str:
        if alias.name == "*":
            return ""
        if alias.asname:
            return alias.asname
        if isinstance(self.statement, ast.Import):
            return alias.name.split(".", maxsplit=1)[0]
        return alias.name

    def alias_import_source(self, alias: ast.alias) -> str:
        alias_source = alias.name
        if alias.asname:
            alias_source = f"{alias.name} as {alias.asname}"
        if isinstance(self.statement, ast.Import):
            return f"import {alias_source}\n"
        module_name = self.statement.module or ""
        module_path = f"{'.' * self.statement.level}{module_name}"
        return f"from {module_path} import {alias_source}\n"


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
