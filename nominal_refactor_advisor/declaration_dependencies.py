"""Lexical dependency projection for movable Python declarations."""

from __future__ import annotations

import ast
import copy
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Callable, TypeAlias

from .annotation_semantics import (
    ModuleNameReferenceScope,
    StringizedAnnotationSurface,
)
from .assignment_projection import (
    NamedAssignmentSelection,
    SingleAssignmentAndValueNameProjection,
)
from .lexical_scopes import (
    ClassNamespaceScope,
    FunctionBindingProjection as FunctionBindingProjection,
    LexicalNameResolution,
    LexicalScopeABC,
    ScopeBindingProjection,
    TypeParameterScope,
)
from .lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ScopeBindingCollector,
    _store_names,
)
from .source_geometry import SourceByteSpan

MovableDeclaration: TypeAlias = (
    ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign
)


def _has_private_identifiers(nodes: tuple[ast.AST, ...]) -> bool:
    names = (
        *(node.id for node in nodes if isinstance(node, ast.Name)),
        *(node.attr for node in nodes if isinstance(node, ast.Attribute)),
        *(node.arg for node in nodes if isinstance(node, ast.arg)),
        *(
            node.name for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ),
    )
    return any(name.startswith("__") and not name.endswith("__") for name in names)


class ClassScopeDependency(StrEnum):
    """Python syntax whose meaning can depend on its enclosing class owner."""

    SUPER_REFERENCE = (
        "super_reference",
        lambda nodes: any(
            isinstance(node, ast.Name) and node.id == "super" for node in nodes
        ),
    )
    CLASS_CELL_REFERENCE = (
        "class_cell_reference",
        lambda nodes: any(
            isinstance(node, ast.Name) and node.id == "__class__" for node in nodes
        ),
    )
    PRIVATE_NAME_MANGLING = ("private_name_mangling", _has_private_identifiers)

    predicate: Callable[[tuple[ast.AST, ...]], bool]

    def __new__(
        cls, value: str, predicate: Callable[[tuple[ast.AST, ...]], bool],
    ) -> "ClassScopeDependency":
        member = str.__new__(cls, value)
        member._value_ = value
        member.predicate = predicate
        return member

    @classmethod
    def from_node(cls, node: ast.AST) -> tuple["ClassScopeDependency", ...]:
        nodes = tuple(ast.walk(node))
        return tuple(dependency for dependency in cls if dependency.predicate(nodes))


class DeclarationDependencyUse(StrEnum):
    """Execution context in which a declaration resolves an external name."""

    EXECUTION = "execution"
    EVALUATED_ANNOTATION = "evaluated_annotation"
    DEFERRED_ANNOTATION = "deferred_annotation"


class ModuleBindingResolutionPhase(StrEnum):
    """Module snapshot from which one direct source reference resolves."""

    SOURCE_POSITION = "source_position", lambda reference: reference.lineno
    FINAL_MODULE = "final_module", lambda _reference: None

    snapshot_line_resolver: Callable[[ast.Name], int | None]

    def __new__(
        cls,
        value: str,
        snapshot_line_resolver: Callable[[ast.Name], int | None],
    ) -> "ModuleBindingResolutionPhase":
        member = str.__new__(cls, value)
        member._value_ = value
        member.snapshot_line_resolver = snapshot_line_resolver
        return member

    def snapshot_line_for(self, reference: ast.Name) -> int | None:
        return self.snapshot_line_resolver(reference)


@dataclass(frozen=True)
class ModuleNameReferenceSurface(ModuleNameReferenceScope):
    """One direct source name and its module-binding evaluation phase."""

    reference: ast.Name
    use: DeclarationDependencyUse
    binding_phase: ModuleBindingResolutionPhase
    resolution: LexicalNameResolution = LexicalNameResolution.EXTERNAL

    @property
    def required_reference(self) -> ast.Name:
        self.resolution.require_known(self.reference.id)
        return self.reference

    @property
    def binding_snapshot_line(self) -> int | None:
        return self.binding_phase.snapshot_line_for(self.reference)

    @property
    def is_direct_annotation(self) -> bool:
        return self.use is DeclarationDependencyUse.EVALUATED_ANNOTATION


@dataclass(frozen=True)
class DeclarationDependencyProjection:
    """External names partitioned by their use in moved declarations."""

    execution_names: frozenset[str]
    evaluated_annotation_names: frozenset[str]
    deferred_annotation_names: frozenset[str]
    annotation_count: int

    @classmethod
    def from_declarations(
        cls,
        declarations: tuple[MovableDeclaration, ...],
    ) -> "DeclarationDependencyProjection":
        collector = _DeclarationDependencyCollector()
        for declaration in declarations:
            collector.visit_declaration(declaration)
        for surface in collector.direct_name_surfaces:
            surface.resolution.require_known(surface.reference.id)
        return cls(
            execution_names=frozenset(
                collector.names_by_use[DeclarationDependencyUse.EXECUTION]
            ),
            evaluated_annotation_names=frozenset(
                collector.names_by_use[DeclarationDependencyUse.EVALUATED_ANNOTATION]
            ),
            deferred_annotation_names=frozenset(
                collector.names_by_use[DeclarationDependencyUse.DEFERRED_ANNOTATION]
            ),
            annotation_count=collector.annotation_count,
        )

    @property
    def annotation_names(self) -> frozenset[str]:
        return self.evaluated_annotation_names | self.deferred_annotation_names

    @property
    def names(self) -> frozenset[str]:
        return self.execution_names | self.annotation_names

    @property
    def annotation_only_names(self) -> frozenset[str]:
        return self.annotation_names - self.execution_names


@dataclass(frozen=True)
class ModuleLexicalDependencyProjection:
    """Reference-bearing dependency surfaces from one lexical traversal."""

    direct_name_surfaces: tuple[ModuleNameReferenceSurface, ...]
    stringized_annotations: tuple[StringizedAnnotationSurface, ...]

    @classmethod
    def require_class_body_independence(cls, node: ast.ClassDef) -> None:
        """Check direct lexical dependencies before removing a class scope.

        Compare references by AST identity, not just name: a lambda's parameter
        may shadow a class field with exactly the same spelling.
        """

        if node.decorator_list or node.keywords or _type_parameter_names(node):
            raise ValueError("class scope header contains unrepresented declarations")
        if any(isinstance(statement, ast.AnnAssign) for statement in node.body):
            raise ValueError("class scope contains annotations not carried by the declaration")
        dependencies = ClassScopeDependency.from_node(node)
        if dependencies:
            raise ValueError(f"class scope dependencies: {', '.join(dependencies)}")
        original = cls.from_module(ast.Module(body=[node], type_ignores=[]))
        flattened = cls.from_module(ast.Module(body=node.body, type_ignores=[]))
        original_references = frozenset(original.external_name_references)
        exposed = tuple(
            reference for reference in flattened.external_name_references
            if reference not in original_references
        )
        if exposed:
            raise ValueError(
                "class scope binds moved references: "
                + ", ".join(sorted({reference.id for reference in exposed}))
            )

    @classmethod
    def from_module(
        cls,
        module: ast.Module,
    ) -> "ModuleLexicalDependencyProjection":
        collector = _DeclarationDependencyCollector()
        for statement in module.body:
            collector.visit(statement)
        return cls(
            direct_name_surfaces=tuple(collector.direct_name_surfaces),
            stringized_annotations=tuple(collector.stringized_annotation_surfaces),
        )

    @property
    def external_name_references(self) -> tuple[ast.Name, ...]:
        return tuple(
            surface.required_reference for surface in self.direct_name_surfaces
        )

    @property
    def direct_annotation_name_surfaces(
        self,
    ) -> tuple[ModuleNameReferenceSurface, ...]:
        return tuple(
            surface
            for surface in self.direct_name_surfaces
            if surface.is_direct_annotation
        )

    def external_surfaces_named(
        self,
        name: str,
    ) -> tuple[ModuleNameReferenceSurface, ...]:
        return tuple(
            surface
            for surface in self.direct_name_surfaces
            if surface.reference.id == name
        )

    def external_references_named(self, name: str) -> tuple[ast.Name, ...]:
        return tuple(
            surface.required_reference for surface in self.external_surfaces_named(name)
        )

    def referenced_names_among(self, names: Iterable[str]) -> frozenset[str]:
        """Project candidate module references from retained source evidence."""

        candidates = frozenset(names)
        direct_names = frozenset(
            surface.reference.id
            for surface in self.direct_name_surfaces
            if surface.reference.id in candidates
        )
        deferred_names = frozenset(
            name
            for name in candidates
            if any(
                surface.reference_count(name)
                and surface.resolves_module_name(name, None)
                for surface in self.stringized_annotations
            )
        )
        return direct_names | deferred_names


@dataclass(frozen=True)
class FunctionBindingABC(ABC):
    """Recover owned reads by removing exactly one declared lexical binding."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    binding_name: str

    @abstractmethod
    def without_binding(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Project the function without this binding, retaining reference identities."""
        raise NotImplementedError

    def required_references(self) -> tuple[ast.Name, ...]:
        without_binding = self.without_binding()
        bindings = FunctionBindingProjection.from_function(without_binding)
        if self.binding_name in (
            bindings.local_names | bindings.global_names | bindings.nonlocal_names
        ) or any(
            isinstance(child, ast.Nonlocal) and self.binding_name in child.names
            for child in ast.walk(self.node)
        ):
            raise ValueError(f"Binding {self.binding_name!r} has additional bindings")
        original_external = frozenset(self.external_references(self.node))
        return tuple(
            reference for reference in self.external_references(without_binding)
            if reference not in original_external
        )

    def external_references(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[ast.Name, ...]:
        return ModuleLexicalDependencyProjection.from_module(
            ast.Module(body=[node], type_ignores=[]),
        ).external_references_named(self.binding_name)


@dataclass(frozen=True)
class FunctionParameterBinding(FunctionBindingABC):
    """One unmodified function-parameter binding."""

    def without_binding(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if self.binding_name not in (
            LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(self.node)
        ):
            raise ValueError(
                f"No parameter {self.binding_name!r} on {self.node.name!r}"
            )
        arguments = copy.copy(self.node.args)
        for field_name, value in ast.iter_fields(arguments):
            if isinstance(value, list):
                setattr(
                    arguments,
                    field_name,
                    [
                        item
                        for item in value
                        if not isinstance(item, ast.arg)
                        or item.arg != self.binding_name
                    ],
                )
            elif isinstance(value, ast.arg) and value.arg == self.binding_name:
                setattr(arguments, field_name, None)
        without_binding = copy.copy(self.node)
        without_binding.args = arguments
        return without_binding


@dataclass(frozen=True)
class FunctionLocalBinding(FunctionBindingABC):
    """One direct, single-name assignment with no other writes to its binding."""

    @cached_property
    def assignment(self) -> ast.stmt:
        (statement,) = NamedAssignmentSelection((self.binding_name,)).statements(
            self.node.body
        )
        return statement

    def without_binding(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        value = SingleAssignmentAndValueNameProjection(self.assignment).value
        if value is None:
            raise ValueError(
                "Local projection requires a single-name assignment with a value"
            )
        without_binding = copy.copy(self.node)
        without_binding.body = [
            ast.Expr(value=value) if statement is self.assignment else statement
            for statement in self.node.body
        ]
        return without_binding

    def required_references(self) -> tuple[ast.Name, ...]:
        references = super().required_references()
        span = SourceByteSpan.require_node(self.assignment)
        if any(
            (read.lineno - 1, read.col_offset) < (span.end_line_index, span.end_byte)
            for read in references
        ):
            raise ValueError(
                f"Local binding {self.binding_name!r} is read before its initializer completes"
            )
        return references


class _DeclarationDependencyCollector(ast.NodeVisitor):
    """Resolve names against the lexical scopes carried by moved declarations."""

    def __init__(self) -> None:
        self.names_by_use: dict[DeclarationDependencyUse, set[str]] = {
            use: set() for use in DeclarationDependencyUse
        }
        self.use = DeclarationDependencyUse.EXECUTION
        self.binding_phase = ModuleBindingResolutionPhase.SOURCE_POSITION
        self.scopes: list[LexicalScopeABC] = []
        self.annotation_count = 0
        self.owner_classes: list[ast.ClassDef] = []
        self.direct_name_surfaces: list[ModuleNameReferenceSurface] = []
        self.stringized_annotation_surfaces: list[StringizedAnnotationSurface] = []

    def visit_declaration(self, node: MovableDeclaration) -> None:
        self.visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._record_class_binding((node.id,), LexicalNameResolution.INTERNAL)
            return
        if isinstance(node.ctx, ast.Del):
            self._record_class_binding((node.id,), LexicalNameResolution.EXTERNAL)
            return
        resolution = self._resolve_name(node.id)
        self._record_reference(node, resolution)

    def _record_reference(
        self,
        node: ast.Name,
        resolution: LexicalNameResolution,
    ) -> None:
        if resolution.is_external_candidate:
            self.names_by_use[self.use].add(node.id)
            if self.use is not DeclarationDependencyUse.DEFERRED_ANNOTATION:
                self.direct_name_surfaces.append(
                    ModuleNameReferenceSurface(
                        owner_classes=tuple(self.owner_classes),
                        reference=node,
                        use=self.use,
                        binding_phase=self.binding_phase,
                        resolution=resolution,
                    )
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_defaults(node.args)
        self.scopes.append(FunctionBindingProjection.from_function(node))
        with self._binding_phase(ModuleBindingResolutionPhase.FINAL_MODULE):
            self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_class(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)
        self._visit_annotation(
            node.annotation,
            evaluation_sensitive=(
                not self.scopes or self._active_class_scope is not None
            ),
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if key is not None:
                self.visit(key)
            self.visit(value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            resolution = self._resolve_name(node.target.id)
            if resolution.is_external_candidate:
                # One token both reads the module and writes the class. A name
                # replacement cannot independently preserve those two owners.
                self._record_reference(node.target, LexicalNameResolution.UNPROVED)
        else:
            self.visit(node.target)
        self.visit(node.value)
        self._record_class_binding(
            _store_names(node.target), LexicalNameResolution.INTERNAL
        )

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self._record_class_binding(
            ImportBoundNameProjection(node).names(), LexicalNameResolution.INTERNAL
        )

    visit_ImportFrom = visit_Import

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        with self._binding_phase(ModuleBindingResolutionPhase.FINAL_MODULE):
            self._visit_comprehension_tail(
                node,
                first,
                remaining,
                (node.elt,),
            )

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
            with self._binding_phase(ModuleBindingResolutionPhase.FINAL_MODULE):
                for statement in node.body:
                    self.visit(statement)
            self.scopes.pop()
        self._record_class_binding((node.name,), LexicalNameResolution.INTERNAL)

    def _visit_class(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        with self._type_parameter_scope(node):
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword)
            self._visit_type_parameters(node)
            scope = ClassNamespaceScope(
                declarations=ScopeBindingProjection.from_nodes(node.body),
            )
            self.scopes.append(scope)
            self.owner_classes.append(node)
            for statement in node.body:
                self.visit(statement)
            self.owner_classes.pop()
            self.scopes.pop()
        self._record_class_binding((node.name,), LexicalNameResolution.INTERNAL)

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
        stringized_surfaces = StringizedAnnotationSurface.from_annotation(
            expression,
            owner_classes=tuple(self.owner_classes),
        )
        self.stringized_annotation_surfaces.extend(stringized_surfaces)
        with self._dependency_use(DeclarationDependencyUse.EVALUATED_ANNOTATION):
            self.visit(expression)
            for surface in stringized_surfaces:
                if surface.expression is not None:
                    self._visit_deferred_annotation(surface.expression)

    def _visit_deferred_annotation(self, expression: ast.expr) -> None:
        """Visit names recursively encoded by one annotation string."""

        with self._dependency_use(DeclarationDependencyUse.DEFERRED_ANNOTATION):
            self.visit(expression)
            for surface in StringizedAnnotationSurface.from_annotation(expression):
                if surface.expression is not None:
                    self._visit_deferred_annotation(surface.expression)

    @contextmanager
    def _type_parameter_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Iterator[None]:
        parameter_names = _type_parameter_names(node)
        if parameter_names:
            self.scopes.append(
                TypeParameterScope(
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
        node: ast.ListComp | ast.SetComp | ast.DictComp,
        result_expressions: tuple[ast.expr, ...],
    ) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        self._visit_comprehension_tail(
            node,
            first,
            remaining,
            result_expressions,
        )

    def _visit_comprehension_tail(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        first: ast.comprehension,
        remaining: list[ast.comprehension],
        result_expressions: tuple[ast.expr, ...],
    ) -> None:
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

    def _resolve_name(self, name: str) -> LexicalNameResolution:
        class_namespace_visible = True
        for scope in reversed(self.scopes):
            resolution = scope.resolve_name(
                name, class_namespace_visible=class_namespace_visible
            )
            if resolution is not None:
                return resolution
            class_namespace_visible &= not scope.hides_enclosing_class_namespace
        return LexicalNameResolution.EXTERNAL

    @property
    def _active_class_scope(self) -> ClassNamespaceScope | None:
        return self.scopes[-1].execution_namespace if self.scopes else None

    def _record_class_binding(
        self,
        names: Iterable[str],
        resolution: LexicalNameResolution,
    ) -> None:
        scope = self._active_class_scope
        if scope is not None:
            scope.record(names, resolution)

    def _visit_nodes(self, nodes: Iterable[ast.AST]) -> None:
        for node in nodes:
            self.visit(node)

    def _visit_alternatives(self, branches: Iterable[Iterable[ast.AST]]) -> None:
        scope = self._active_class_scope
        if scope is None:
            for branch in branches:
                self._visit_nodes(branch)
            return
        before = dict(scope.bindings)
        alternatives = []
        for branch in branches:
            scope.bindings = dict(before)
            self._visit_nodes(branch)
            alternatives.append(scope.bindings)
        scope.join(alternatives)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_alternatives((node.body, node.orelse))

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self._visit_alternatives(((node.body,), (node.orelse,)))

    def _visit_short_circuit(self, expressions: list[ast.expr]) -> None:
        alternatives = []
        scope = self._active_class_scope
        for expression in expressions:
            self.visit(expression)
            if scope is not None:
                alternatives.append(dict(scope.bindings))
        if scope is not None:
            scope.join(alternatives)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._visit_short_circuit(node.values)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        self._visit_short_circuit(node.comparators)

    def _visit_unproved_control_flow(self, node: ast.AST) -> None:
        """Retain uncertainty for effects needing loop or exception path proofs."""

        scope = self._active_class_scope
        if scope is None:
            self.generic_visit(node)
            return
        bindings = ScopeBindingCollector()
        bindings.visit(node)
        with scope.unproved_execution(bindings.bound_names):
            self.generic_visit(node)

    visit_For = _visit_unproved_control_flow
    visit_AsyncFor = _visit_unproved_control_flow
    visit_While = _visit_unproved_control_flow
    visit_Try = _visit_unproved_control_flow
    visit_TryStar = _visit_unproved_control_flow
    visit_With = _visit_unproved_control_flow
    visit_AsyncWith = _visit_unproved_control_flow
    visit_Match = _visit_unproved_control_flow

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

    @contextmanager
    def _binding_phase(
        self,
        phase: ModuleBindingResolutionPhase,
    ) -> Iterator[None]:
        previous = self.binding_phase
        self.binding_phase = phase
        try:
            yield
        finally:
            self.binding_phase = previous


def _type_parameter_names(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    if sys.version_info < (3, 12):
        return frozenset()
    return frozenset(parameter.name for parameter in node.type_params)
