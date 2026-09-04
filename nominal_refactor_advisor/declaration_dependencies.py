"""Lexical dependency projection for movable Python declarations."""

from __future__ import annotations

import ast
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterator, TypeAlias

from .ast_tools import ImportBoundNameProjection

MovableDeclaration: TypeAlias = (
    ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign
)


class DeclarationDependencyUse(StrEnum):
    """Execution context in which a declaration resolves an external name."""

    EXECUTION = "execution"
    ANNOTATION = "annotation"


@dataclass(frozen=True)
class DeclarationDependencyProjection:
    """External names partitioned by their use in moved declarations."""

    execution_names: frozenset[str]
    annotation_names: frozenset[str]
    annotation_count: int

    @classmethod
    def from_declarations(
        cls,
        declarations: tuple[MovableDeclaration, ...],
    ) -> "DeclarationDependencyProjection":
        collector = _DeclarationDependencyCollector()
        for declaration in declarations:
            collector.visit_declaration(declaration)
        return cls(
            execution_names=frozenset(
                collector.names_by_use[DeclarationDependencyUse.EXECUTION]
            ),
            annotation_names=frozenset(
                collector.names_by_use[DeclarationDependencyUse.ANNOTATION]
            ),
            annotation_count=collector.annotation_count,
        )

    @property
    def names(self) -> frozenset[str]:
        return self.execution_names | self.annotation_names

    @property
    def annotation_only_names(self) -> frozenset[str]:
        return self.annotation_names - self.execution_names


@dataclass(frozen=True)
class FunctionBindingProjection:
    """Bindings owned by one function's compile-time lexical scope."""

    local_names: frozenset[str]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]

    @classmethod
    def from_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> "FunctionBindingProjection":
        collector = _CurrentScopeBindingCollector()
        if isinstance(node, ast.Lambda):
            collector.visit(node.body)
        else:
            for statement in node.body:
                collector.visit(statement)
        local_names = collector.bound_names | _argument_names(node.args)
        return cls(
            local_names=frozenset(
                local_names - collector.global_names - collector.nonlocal_names
            ),
            global_names=frozenset(collector.global_names),
            nonlocal_names=frozenset(collector.nonlocal_names),
        )


@dataclass
class _ClassScope:
    available_names: set[str] = field(default_factory=set)
    global_names: frozenset[str] = frozenset()
    nonlocal_names: frozenset[str] = frozenset()


_DependencyScope: TypeAlias = FunctionBindingProjection | _ClassScope


@dataclass
class _ComprehensionContainingScopeBindingCollector(ast.NodeVisitor):
    """Collect walrus targets owned outside comprehension scopes."""

    bound_names: set[str]

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.bound_names.update(_store_names(node.target))
        self.visit(node.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _CurrentScopeBindingCollector(ast.NodeVisitor):
    """Collect compile-time bindings without descending into child scopes."""

    def __init__(self) -> None:
        self.bound_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

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
        _ComprehensionContainingScopeBindingCollector(self.bound_names).visit(node)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class _DeclarationDependencyCollector(ast.NodeVisitor):
    """Resolve names against the lexical scopes carried by moved declarations."""

    def __init__(self) -> None:
        self.names_by_use: dict[DeclarationDependencyUse, set[str]] = {
            use: set() for use in DeclarationDependencyUse
        }
        self.use = DeclarationDependencyUse.EXECUTION
        self.scopes: list[_DependencyScope] = []
        self.annotation_count = 0

    def visit_declaration(self, node: MovableDeclaration) -> None:
        self.visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_internal(node.id):
            self.names_by_use[self.use].add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_defaults(node.args)
        self.scopes.append(FunctionBindingProjection.from_function(node))
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_class(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotation(
            node.annotation,
            evaluation_sensitive=(
                not self.scopes or isinstance(self.scopes[-1], _ClassScope)
            ),
        )
        if node.value is not None:
            self.visit(node.value)
            if not isinstance(node.target, ast.Name):
                self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            if not self._is_internal(node.target.id):
                self.names_by_use[self.use].add(node.target.id)
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_Import(self, node: ast.Import) -> None:
        return

    visit_ImportFrom = visit_Import

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_defaults(node.args)
        with self._type_parameter_scope(node):
            self._visit_type_parameters(node)
            self._visit_argument_annotations(node.args)
            if node.returns is not None:
                self._visit_annotation(node.returns)
            self.scopes.append(FunctionBindingProjection.from_function(node))
            for statement in node.body:
                self.visit(statement)
            self.scopes.pop()

    def _visit_class(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        with self._type_parameter_scope(node):
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword)
            self._visit_type_parameters(node)
            bindings = _CurrentScopeBindingCollector()
            for statement in node.body:
                bindings.visit(statement)
            scope = _ClassScope(
                global_names=frozenset(bindings.global_names),
                nonlocal_names=frozenset(bindings.nonlocal_names),
            )
            self.scopes.append(scope)
            for statement in node.body:
                self.visit(statement)
                self._apply_class_binding(statement, scope)
            self.scopes.pop()

    def _visit_argument_defaults(self, arguments: ast.arguments) -> None:
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_argument_annotations(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is not None:
                self._visit_annotation(argument.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self._visit_annotation(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self._visit_annotation(arguments.kwarg.annotation)

    def _visit_type_parameters(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        if sys.version_info < (3, 12):
            return
        for parameter in node.type_params:
            for expression in ast.iter_child_nodes(parameter):
                if isinstance(expression, ast.expr):
                    self._visit_annotation(expression)

    def _visit_annotation(
        self,
        expression: ast.expr,
        *,
        evaluation_sensitive: bool = True,
    ) -> None:
        if evaluation_sensitive:
            self.annotation_count += 1
        with self._dependency_use(DeclarationDependencyUse.ANNOTATION):
            self.visit(expression)

    @contextmanager
    def _type_parameter_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Iterator[None]:
        parameter_names = _type_parameter_names(node)
        if parameter_names:
            self.scopes.append(
                FunctionBindingProjection(
                    local_names=parameter_names,
                    global_names=frozenset(),
                    nonlocal_names=frozenset(),
                )
            )
        try:
            yield
        finally:
            if parameter_names:
                self.scopes.pop()

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        result_expressions: tuple[ast.expr, ...],
    ) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        local_names = {
            name
            for generator in node.generators
            for name in _store_names(generator.target)
        }
        self.scopes.append(
            FunctionBindingProjection(
                local_names=frozenset(local_names),
                global_names=frozenset(),
                nonlocal_names=frozenset(),
            )
        )
        for condition in first.ifs:
            self.visit(condition)
        for generator in remaining:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in result_expressions:
            self.visit(expression)
        self.scopes.pop()

    def _is_internal(self, name: str) -> bool:
        crossed_function_scope = False
        crossed_class_scope = False
        for scope in reversed(self.scopes):
            if isinstance(scope, FunctionBindingProjection):
                if name in scope.global_names:
                    return False
                if name in scope.local_names:
                    return True
                crossed_function_scope = True
                continue
            if name in scope.global_names:
                return False
            if (
                not crossed_function_scope
                and not crossed_class_scope
                and name in scope.available_names
            ):
                return True
            crossed_class_scope = True
        return False

    @staticmethod
    def _apply_class_binding(statement: ast.stmt, scope: _ClassScope) -> None:
        bound_names: set[str] = set()
        removed_names: set[str] = set()
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                bound_names.update(_store_names(target))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            bound_names.update(_store_names(statement.target))
        elif isinstance(statement, ast.AugAssign):
            bound_names.update(_store_names(statement.target))
        elif isinstance(statement, ast.Import | ast.ImportFrom):
            bound_names.update(ImportBoundNameProjection(statement).names())
        elif isinstance(
            statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            bound_names.add(statement.name)
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                removed_names.update(_store_names(target))
        scope.available_names.difference_update(removed_names)
        scope.available_names.update(
            bound_names - scope.global_names - scope.nonlocal_names
        )

    @contextmanager
    def _dependency_use(
        self,
        use: DeclarationDependencyUse,
    ) -> Iterator[None]:
        previous = self.use
        self.use = use
        try:
            yield
        finally:
            self.use = previous


def _argument_names(arguments: ast.arguments) -> set[str]:
    return {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *((arguments.vararg,) if arguments.vararg is not None else ()),
            *((arguments.kwarg,) if arguments.kwarg is not None else ()),
        )
    }


def _type_parameter_names(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    if sys.version_info < (3, 12):
        return frozenset()
    return frozenset(parameter.name for parameter in node.type_params)


def _store_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(name for element in target.elts for name in _store_names(element))
    return ()
