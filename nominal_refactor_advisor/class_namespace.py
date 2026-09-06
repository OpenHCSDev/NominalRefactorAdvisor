"""Source evidence for class namespace bindings and definition-time execution."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from types import ClassMethodDescriptorType
from typing import ClassVar, Iterator, TYPE_CHECKING

from .ast_tools import ModuleAnnotationEvaluationMode
from .ast_projection import AstExpressionProjection
from .declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleBindingResolutionPhase,
    _DeclarationDependencyCollector,
)
from .lexical_scopes import (
    ClassNamespaceScope,
    LexicalNameResolution,
    LexicalScopeABC,
    LexicalScopeContext,
)
from .native_declarations import NativeDeclaration

if TYPE_CHECKING:
    from .ast_tools import ParsedModule
    from .class_index import ModuleNominalBindingView


# Native descriptor construction does not execute the wrapped function or expose
# the caller's namespace. Names and import spellings derive from these types.
NATIVE_METHOD_DECORATORS = (classmethod, property, staticmethod)


@dataclass(frozen=True)
class ClassNamespaceEffect(ABC):
    node: ast.AST
    use: DeclarationDependencyUse

    @classmethod
    def from_scope(
        cls, node: ast.AST, use: DeclarationDependencyUse, scope: LexicalScopeContext
    ) -> ClassNamespaceEffect:
        return cls(node, use)

    def executes_at_definition(self, module: ParsedModule) -> bool:
        return (
            self.use.binding_phase(
                ModuleBindingResolutionPhase.SOURCE_POSITION,
                eager_annotations=ModuleAnnotationEvaluationMode.from_module(
                    module.module
                ).annotations_execute_at_declaration,
            )
            is ModuleBindingResolutionPhase.SOURCE_POSITION
        )

    @abstractmethod
    def require_closed(
        self,
        bindings: ModuleNominalBindingView,
        module: ParsedModule,
        owner: ast.ClassDef,
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class NativeClassNamespaceEffect(ClassNamespaceEffect, ABC):
    node: ast.expr
    root_resolution: LexicalNameResolution
    native_declarations: ClassVar[tuple[type, ...]]

    @classmethod
    def from_scope(
        cls, node: ast.expr, use: DeclarationDependencyUse, scope: LexicalScopeContext
    ) -> NativeClassNamespaceEffect:
        chain = AstExpressionProjection.attribute_chain(node)
        return cls(
            node,
            use,
            (
                LexicalNameResolution.UNPROVED
                if chain is None
                else scope._resolve_name(chain[0])
            ),
        )

    def require_closed(
        self,
        bindings: ModuleNominalBindingView,
        module: ParsedModule,
        owner: ast.ClassDef,
    ) -> None:
        if self.root_resolution is not LexicalNameResolution.EXTERNAL:
            raise ValueError(
                f"Class namespace execution at line {self.node.lineno} has no external binding proof"
            )
        witness = bindings.reference_or_builtin_witness_at(
            module,
            self.node,
            line=owner.lineno,
        )
        if witness is None or not any(
            witness.qualified_name == NativeDeclaration(native).qualified_name
            for native in self.native_declarations
        ):
            raise ValueError(
                f"Class namespace execution at line {self.node.lineno} remains unproved"
            )


class DescriptorClassNamespaceEffect(NativeClassNamespaceEffect):
    native_declarations = NATIVE_METHOD_DECORATORS


class GenericAliasClassNamespaceEffect(NativeClassNamespaceEffect):
    # Builtin generic alias construction stores its arguments without executing
    # repository subscription implementations. Custom __class_getitem__ stays open.
    native_declarations = tuple(
        declaration
        for declaration in vars(builtins).values()
        if isinstance(declaration, type)
        and "__class_getitem__" in vars(declaration)
        and isinstance(
            vars(declaration)["__class_getitem__"], ClassMethodDescriptorType
        )
    )


class InstalledClassNamespaceValue(ClassNamespaceEffect):
    def require_closed(
        self,
        bindings: ModuleNominalBindingView,
        module: ParsedModule,
        owner: ast.ClassDef,
    ) -> None:
        del bindings, module, owner
        if isinstance(self.node, (ast.Call, ast.Lambda, ast.Tuple, ast.List)):
            # Calls/defaults have separate evidence. Native sequence construction
            # has no element hashing or class installation hooks.
            return
        try:
            ast.literal_eval(self.node)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(
                f"Class namespace value at line {self.node.lineno} may have creation hooks"
            ) from error


@dataclass(frozen=True)
class ClassNamespaceExecutionEvidence:
    """Final lexical bindings and execution obligations from one scope traversal."""

    binding_names: frozenset[str]
    effects: tuple[ClassNamespaceEffect, ...]

    @classmethod
    def from_class(cls, owner: ast.ClassDef) -> ClassNamespaceExecutionEvidence:
        collector = _ClassNamespaceExecutionCollector(owner)
        collector.visit(owner)
        return cls(
            frozenset(
                name
                for name, resolution in collector.completed_scope.bindings.items()
                if resolution is not LexicalNameResolution.EXTERNAL
            ),
            tuple(collector.effect_projection.effects_by_node.values()),
        )

    def require_closed(
        self,
        bindings: ModuleNominalBindingView,
        module: ParsedModule,
        owner: ast.ClassDef,
    ) -> None:
        for effect in self.effects:
            if effect.executes_at_definition(module):
                effect.require_closed(bindings, module, owner)


class _ClassNamespaceEffectProjection(ast.NodeVisitor):
    """Select one node's effects; the scope collector alone traverses children."""

    def __init__(self, scope: _DeclarationDependencyCollector) -> None:
        self.scope = scope
        self.effects_by_node: dict[ast.AST, ClassNamespaceEffect] = {}

    def _record_effect(
        self, effect_type: type[ClassNamespaceEffect], node: ast.AST
    ) -> None:
        self.effects_by_node.setdefault(
            node, effect_type.from_scope(node, self.scope.use, self.scope)
        )

    def generic_visit(self, node: ast.AST) -> None:
        # Unknown executable forms require proof rather than implicit trust.
        if isinstance(node, (ast.expr, ast.stmt)):
            self._record_effect(InstalledClassNamespaceValue, node)

    def visit_Name(self, node: ast.Name) -> None:
        # Lookup and binding are accounted for by the lexical traversal.
        pass

    visit_Pass = visit_Delete = visit_Lambda = visit_Tuple = visit_List = visit_Name

    def visit_Call(self, node: ast.Call) -> None:
        self._record_effect(DescriptorClassNamespaceEffect, node.func)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._record_effect(GenericAliasClassNamespaceEffect, node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self._record_effect(DescriptorClassNamespaceEffect, decorator)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(
        self, node: ast.Assign | ast.AnnAssign | ast.NamedExpr | ast.Expr
    ) -> None:
        if node.value is not None:
            self._record_effect(InstalledClassNamespaceValue, node.value)

    visit_AnnAssign = visit_NamedExpr = visit_Expr = visit_Assign

    def visit_If(self, node: ast.If | ast.IfExp) -> None:
        self._record_effect(InstalledClassNamespaceValue, node.test)

    visit_IfExp = visit_If

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        # Python obtains the outer iterator immediately; its body is deferred.
        self._record_effect(InstalledClassNamespaceValue, node.generators[0].iter)


class _ClassNamespaceExecutionCollector(_DeclarationDependencyCollector):
    """Attach effect evidence to the existing ordered, scope-aware traversal."""

    completed_scope: ClassNamespaceScope

    def __init__(self, owner: ast.ClassDef) -> None:
        super().__init__()
        self.owner = owner
        self.effect_projection = _ClassNamespaceEffectProjection(self)

    def visit(self, node: ast.AST) -> None:
        if (
            self.owner in self.owner_classes
            and self.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
        ):
            self.effect_projection.visit(node)
        super().visit(node)

    @contextmanager
    def _scope(self, scope: LexicalScopeABC) -> Iterator[None]:
        with super()._scope(scope):
            yield
        if isinstance(scope, ClassNamespaceScope) and scope.node is self.owner:
            self.completed_scope = scope
