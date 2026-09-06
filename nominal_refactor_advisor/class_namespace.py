"""Source evidence for class namespace bindings and definition-time execution."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Iterator, TYPE_CHECKING

from .ast_tools import ModuleAnnotationEvaluationMode
from .declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleBindingResolutionPhase,
    _DeclarationDependencyCollector,
)
from .descriptor_algebra import AliasProperty
from .lexical_scopes import (
    ClassNamespaceScope,
    LexicalNameResolution,
    LexicalScopeABC,
    LexicalScopeContext,
)
from .native_declarations import NativeDeclaration
from .native_reference import (
    NativeArgumentEvidence,
    NativeReferenceEnvironment,
    ScopedNativeReference,
)
from .native_subscription import NativeArgumentInspection, NativeSubscriptionAuthority

if TYPE_CHECKING:
    from .ast_tools import ParsedModule
    from .class_index import ModuleNominalBindingView


# Native descriptor decorators receive the function just declared. Explicit
# construction also requires evidence about the argument's metadata access.
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

    @property
    def recording_node(self) -> ast.AST:
        return self.node

    @abstractmethod
    def require_closed(
        self,
        environment: NativeReferenceEnvironment,
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ReferencedClassNamespaceEffect(ClassNamespaceEffect, ABC):
    node: ast.expr
    reference: ScopedNativeReference

    recording_node = AliasProperty[ast.AST]("reference.node")


class NativeClassNamespaceEffect(ReferencedClassNamespaceEffect, ABC):
    native_declarations: ClassVar[tuple[NativeDeclaration, ...]]

    @classmethod
    def from_scope(
        cls, node: ast.expr, use: DeclarationDependencyUse, scope: LexicalScopeContext
    ) -> NativeClassNamespaceEffect:
        return cls(node, use, ScopedNativeReference.from_scope(node, scope))

    def require_closed(
        self,
        environment: NativeReferenceEnvironment,
    ) -> None:
        self.reference.require_native(environment, self.native_declarations)


class DescriptorClassNamespaceEffect(NativeClassNamespaceEffect):
    native_declarations = tuple(
        NativeDeclaration(native) for native in NATIVE_METHOD_DECORATORS
    )


@dataclass(frozen=True)
class DescriptorCallClassNamespaceEffect(DescriptorClassNamespaceEffect):
    node: ast.Call
    arguments: tuple[NativeArgumentEvidence, ...]

    @classmethod
    def from_scope(
        cls, node: ast.Call, use: DeclarationDependencyUse, scope: LexicalScopeContext
    ) -> DescriptorCallClassNamespaceEffect:
        return cls(
            node,
            use,
            ScopedNativeReference.from_scope(node.func, scope),
            tuple(
                NativeArgumentEvidence.from_scope(argument, scope)
                for argument in (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                )
            ),
        )

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        super().require_closed(environment)
        for argument in self.arguments:
            NativeArgumentInspection(argument, environment).visit(argument.node)


@dataclass(frozen=True)
class SubscriptionClassNamespaceEffect(ReferencedClassNamespaceEffect):
    node: ast.Subscript
    argument: NativeArgumentEvidence

    @classmethod
    def from_scope(
        cls,
        node: ast.Subscript,
        use: DeclarationDependencyUse,
        scope: LexicalScopeContext,
    ) -> SubscriptionClassNamespaceEffect:
        return cls(
            node,
            use,
            ScopedNativeReference.from_scope(node.value, scope),
            NativeArgumentEvidence.from_scope(node.slice, scope),
        )

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        NativeSubscriptionAuthority.for_reference(
            self.reference, environment
        ).require_argument(self.argument, environment)


class InstalledClassNamespaceValue(ClassNamespaceEffect):
    def require_closed(
        self,
        environment: NativeReferenceEnvironment,
    ) -> None:
        del environment
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
        environment = NativeReferenceEnvironment(bindings, module, owner.lineno)
        for effect in self.effects:
            if effect.executes_at_definition(module):
                effect.require_closed(environment)


class _ClassNamespaceEffectProjection(ast.NodeVisitor):
    """Select one node's effects; the scope collector alone traverses children."""

    def __init__(self, scope: _DeclarationDependencyCollector) -> None:
        self.scope = scope
        self.effects_by_node: dict[ast.AST, ClassNamespaceEffect] = {}

    def _record_effect(
        self, effect_type: type[ClassNamespaceEffect], node: ast.AST
    ) -> None:
        effect = effect_type.from_scope(node, self.scope.use, self.scope)
        self.effects_by_node.setdefault(effect.recording_node, effect)

    def generic_visit(self, node: ast.AST) -> None:
        # Unknown executable forms require proof rather than implicit trust.
        if isinstance(node, (ast.expr, ast.stmt)):
            self._record_effect(InstalledClassNamespaceValue, node)

    def visit_Name(self, node: ast.Name) -> None:
        # Lookup and binding are accounted for by the lexical traversal.
        pass

    visit_Pass = visit_Delete = visit_Lambda = visit_Tuple = visit_List = visit_Name

    def visit_Call(self, node: ast.Call) -> None:
        self._record_effect(DescriptorCallClassNamespaceEffect, node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._record_effect(SubscriptionClassNamespaceEffect, node)

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
