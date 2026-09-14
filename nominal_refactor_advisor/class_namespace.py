"""Source evidence for class namespace bindings and definition-time execution."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import (
    ClassVar,
    TYPE_CHECKING,
    cast,
)

from .ast_tools import ModuleAnnotationEvaluationMode
from .captured_reference import SingleFlowPrefix
from .declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleBindingResolutionPhase,
    _DeclarationDependencyCollector,
)
from .lexical_scopes import (
    LexicalNameResolution,
    LexicalScopeABC,
)
from .native_compilation import NativeCreationBackend
from .native_declarations import NativeDeclaration
from .native_call import NativeDescriptorCall
from .product_flow import (
    CompactCallableReferenceUse,
    CompactDefinitionTarget,
    CompactEvaluatedResult,
    CompactFlowPosition,
    CompactFunctionCall,
    CompactMutation,
    CompactSubscription,
    CompactValueUse,
    SourceFlowEvaluation,
    SourceFlowEvent,
    SourceFlowOperation,
    SourceProductFlowProjection,
)
from .native_reference import NativeReferenceEnvironment
from .value_expression import LiteralExpressionEffects as LiteralExpressionEffects

if TYPE_CHECKING:
    from .ast_tools import ParsedModule
    from .class_index import ModuleNominalBindingView


# Native descriptor decorators receive the function just declared. Explicit
# construction also requires evidence about the argument's metadata access.
NATIVE_METHOD_DECORATORS = tuple(
    native.declaration for native in NativeDescriptorCall.native_declarations
)


@dataclass(frozen=True)
class ClassNamespaceEffect(ABC):
    node: ast.AST
    use: DeclarationDependencyUse

    application_event_types: ClassVar[tuple[type[SourceFlowEvent], ...]] = ()

    def earliest_application(
        self,
        source: SourceProductFlowProjection,
        evaluation: SourceFlowEvaluation,
    ) -> CompactFlowPosition:
        """Unresolved effects retain the original whole-expression observer bound."""
        return evaluation.entry

    def require_operation(
        self,
        environment: NativeReferenceEnvironment,
        operation: SourceFlowOperation,
    ) -> None:
        """Close an occurrence; atomic effects retain their existing node obligation."""

        self.require_closed(environment)

    def application_operations(
        self, site: SourceEffectSite, source: SourceProductFlowProjection
    ) -> tuple[SourceFlowOperation, ...] | None:
        """Select actual operations, or retain an unresolved observer interval.

        None means no point-event proof. An exhaustive projection may instead
        return an empty tuple, proving there are no operations for this effect.
        """
        return (
            tuple(
                operation
                for operation in source.operations_by_node.get(site.trigger, ())
                if isinstance(operation.event, self.application_event_types)
            )
            or None
        )

    def executes_at_definition(
        self,
        annotation_mode: ModuleAnnotationEvaluationMode,
    ) -> bool:
        return (
            self.use.binding_phase(
                ModuleBindingResolutionPhase.SOURCE_POSITION,
                eager_annotations=annotation_mode.annotations_execute_at_declaration,
            )
            is ModuleBindingResolutionPhase.SOURCE_POSITION
        )

    @abstractmethod
    def require_closed(
        self,
        environment: NativeReferenceEnvironment,
    ) -> None:
        raise NotImplementedError


class NativeClassNamespaceEffect(ClassNamespaceEffect, ABC):
    node: ast.expr
    native_declarations: ClassVar[tuple[NativeDeclaration, ...]]
    application_event_types = (CompactMutation,)

    @property
    def operand(self) -> ast.expr:
        return self.node

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_native(self.operand, self.native_declarations)


class DescriptorClassNamespaceEffect(NativeClassNamespaceEffect):
    native_declarations = NativeDescriptorCall.native_declarations


@dataclass(frozen=True)
class CallClassNamespaceEffect(ClassNamespaceEffect):
    node: ast.Call
    application_event_types = (CompactFunctionCall,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_call(self.node)


@dataclass(frozen=True)
class SubscriptionClassNamespaceEffect(ClassNamespaceEffect):
    node: ast.Subscript

    application_event_types = (CompactSubscription,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_subscription(self.node)


class ReferenceAccessEffect(ClassNamespaceEffect):
    node: ast.Attribute
    application_event_types = (CompactCallableReferenceUse,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.capture(self.node).require_closed()


class NamespaceWriteEffect(ClassNamespaceEffect):
    node: ast.Attribute
    application_event_types = (CompactMutation,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_namespace_write(self.node)


class ItemWriteEffect(ClassNamespaceEffect):
    node: ast.Subscript
    application_event_types = (CompactMutation,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_item_write(self.node)


class BindingSourceEffect(ClassNamespaceEffect):
    application_event_types = (CompactMutation,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_binding_write(self.node)


class DefinitionHeaderEffect(ClassNamespaceEffect):
    """Consume header inputs at their original evaluation cuts, before binding."""

    def application_operations(
        self, site: SourceEffectSite, source: SourceProductFlowProjection
    ) -> tuple[SourceFlowOperation, ...]:
        definition: CompactMutation[CompactDefinitionTarget] = cast(
            CompactMutation,
            source.definition_operation(self.node).event,
        )
        return tuple(
            source.event_operation(use) for use in definition.target.header_uses
        )

    def require_operation(
        self,
        environment: NativeReferenceEnvironment,
        operation: SourceFlowOperation,
    ) -> None:
        source = environment.source
        definition = source.definition_operation(self.node)
        target: CompactDefinitionTarget = cast(CompactMutation, definition.event).target
        use = operation.event
        if not isinstance(use, CompactValueUse) or not any(
            original is use for original in target.header_uses
        ):
            raise ValueError("Definition input requires its original header receipt")
        context = source.context_for_owner(definition.owner)
        if source.source_operation(context, use) is not operation:
            raise ValueError("Definition input requires its original source operation")
        environment.kernel._read_use(use, context, frozenset()).require_closed()

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        raise ValueError("Definition headers require their original input operations")


class ExpressionStatementEffect(ClassNamespaceEffect):
    """Consume the original value disposition, without rediscovering syntax."""

    node: ast.Expr

    def result(self, source: SourceProductFlowProjection) -> CompactEvaluatedResult:
        results = tuple(
            operation.event
            for operation in source.operations_by_node.get(self.node, ())
            if isinstance(operation.event, CompactEvaluatedResult)
        )
        if len(results) != 1:
            raise ValueError("Expression has no unique original evaluated result")
        return results[0]

    def application_operations(
        self,
        site: SourceEffectSite,
        source: SourceProductFlowProjection,
    ) -> tuple[SourceFlowOperation, ...] | None:
        result = self.result(source)
        return (
            result.destination.use.expression_operations(
                result,
                source.operations_by_node.get(site.trigger, ()),
            )
            or None
        )

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        self.result(environment.source).destination.use.require_expression(
            environment, self.node
        )


class AssignmentSourceEffect(ClassNamespaceEffect):
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr
    application_event_types = (CompactEvaluatedResult,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_assignment(self.node)


class AnnotationStorageEffect(ItemWriteEffect):
    """An annotation's implicit item write, distinct from evaluating its value."""

    node: ast.AnnAssign

    def executes_at_definition(
        self, annotation_mode: ModuleAnnotationEvaluationMode
    ) -> bool:
        return (
            annotation_mode.stores_variable_annotations_at_definition
            and super().executes_at_definition(annotation_mode)
        )


class ImportSourceEffect(ClassNamespaceEffect):
    node: ast.Import | ast.ImportFrom
    application_event_types = (CompactMutation,)

    def require_operation(
        self,
        environment: NativeReferenceEnvironment,
        operation: SourceFlowOperation,
    ) -> None:
        environment.require_import_operation(operation)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_import(self.node)


class ClassCreationEffect(ClassNamespaceEffect):
    node: ast.ClassDef
    application_event_types = (CompactMutation,)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        environment.require_class_creation(self.node)


class DictionaryConstructionEffect(ClassNamespaceEffect):
    """Native dictionary assembly; child expressions retain their own effects."""

    node: ast.Dict

    def earliest_application(
        self,
        source: SourceProductFlowProjection,
        evaluation: SourceFlowEvaluation,
    ) -> CompactFlowPosition:
        """Refine only the earliest possible effect, not native assembly timing."""
        if evaluation.node is not self.node:
            raise ValueError("Dictionary effect requires its original evaluation")
        operand = (
            NativeCreationBackend.current().dictionary_construction_initial_operand(
                self.node
            )
        )
        if operand is None:
            return super().earliest_application(source, evaluation)
        return source.evaluation_operand_entry(evaluation, operand)

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        NativeCreationBackend.current().require_dictionary_construction(self.node)


class LiteralSourceEffect(ClassNamespaceEffect):
    """Literal evaluation and truth cannot dispatch to source-defined values."""

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        LiteralExpressionEffects(self.node).require_closed()


class LiteralIterationEffect(ClassNamespaceEffect):
    """Iterator acquisition is distinct from constructing or testing a value."""

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        value = LiteralExpressionEffects(self.node).value
        try:
            iter(value)
        except TypeError as error:
            raise ValueError("Literal iterator acquisition remains unproved") from error


@dataclass(frozen=True)
class SourceEffectSite:
    """One classifier invocation and its distinct execution obligations.

    The trigger identifies the operation whose effects need closure. Operands
    remain on individual effects: a decorator reference is captured before the
    definition trigger applies it. A source consumer must join this trigger to
    actual operation/interval evidence for each declared effect. Header input
    consumption belongs at its operand receipt; decorator application belongs
    at the definition's application event. This relation alone makes no
    execution-order or activation claim.
    """

    trigger: ast.AST
    effects: tuple[ClassNamespaceEffect, ...]
    scope_path: tuple[LexicalScopeABC, ...]
    binding_phase: ModuleBindingResolutionPhase


@dataclass(frozen=True)
class SourceEffectOccurrence(ABC):
    """An original effect selected in one actual contextual execution interval."""

    effect: ClassNamespaceEffect
    interval: SingleFlowPrefix

    @abstractmethod
    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class OperationEffectOccurrence(SourceEffectOccurrence):
    """Apply the effect at its original operation, including per-alias imports."""

    operation: SourceFlowOperation

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        self.effect.require_operation(environment, self.operation)


@dataclass(frozen=True)
class EvaluationEffectOccurrence(SourceEffectOccurrence):
    """Retain an unresolved observer interval, not a fabricated application event."""

    evaluation: SourceFlowEvaluation

    def require_closed(self, environment: NativeReferenceEnvironment) -> None:
        self.effect.require_closed(environment)


@dataclass(frozen=True)
class SourceExecutionEffectEvidence:
    """Effects and completed lexical scopes from one existing source traversal.

    Scope ownership is not activation. In particular a type-parameter scope's
    source owner need not identify a native executing frame. Consumers join
    triggers to retained source-flow operations before proving execution cuts.
    """

    sites: tuple[SourceEffectSite, ...]
    completed_scopes: tuple[LexicalScopeABC, ...]

    def occurrences(
        self, source: SourceProductFlowProjection, interval: SingleFlowPrefix
    ) -> Iterator[SourceEffectOccurrence]:
        """Select original obligations once for every consumer of execution effects."""
        owner = source.source_nodes_by_owner[id(interval.context.flow.owner)]
        for site, effect in self.definition_effects(owner, source.module):
            operations = effect.application_operations(site, source)
            if operations is not None:
                for operation in operations:
                    if (
                        operation.owner is interval.context.flow.owner
                        and interval.contains(operation.event)
                    ):
                        yield OperationEffectOccurrence(effect, interval, operation)
                continue
            evaluations = tuple(
                evaluation
                for evaluation in source.evaluation_bounds_by_node.get(site.trigger, ())
                if evaluation.owner is interval.context.flow.owner
            )
            if not evaluations:
                raise ValueError(
                    "Source effect has no actual operation or evaluation bound"
                )
            for evaluation in evaluations:
                if interval.may_overlap_positions(
                    effect.earliest_application(source, evaluation), evaluation.exit
                ):
                    yield EvaluationEffectOccurrence(effect, interval, evaluation)

    def definition_effects(
        self,
        owner: ast.AST,
        module: ParsedModule,
    ) -> tuple[tuple[SourceEffectSite, ClassNamespaceEffect], ...]:
        """Project current-definition effects; retain deferred sites in the inventory."""
        annotation_mode = ModuleAnnotationEvaluationMode.from_module(module.module)
        return tuple(
            (site, effect)
            for site in self.sites_by_owner.get(owner, ())
            for effect in site.effects
            if effect.executes_at_definition(annotation_mode)
        )

    @classmethod
    def from_source(
        cls, root: ast.Module | ast.ClassDef
    ) -> SourceExecutionEffectEvidence:
        collector = _SourceExecutionEffectCollector()
        collector.visit(root)
        return cls(tuple(collector.effect_sites), tuple(collector.completed_scopes))

    @cached_property
    def sites_by_owner(self) -> dict[ast.AST, tuple[SourceEffectSite, ...]]:
        grouped: dict[ast.AST, list[SourceEffectSite]] = {}
        for site in self.sites:
            grouped.setdefault(site.scope_path[-1].node, []).append(site)
        return {owner: tuple(sites) for owner, sites in grouped.items()}

    def class_evidence(self, owner: ast.ClassDef) -> ClassNamespaceExecutionEvidence:
        if any(
            site.trigger is owner
            and site.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
            for site in self.sites
        ):
            raise ValueError(
                "Class facade requires a source-position entry, not deferred activation"
            )
        namespaces = tuple(
            namespace
            for scope in self.completed_scopes
            if (namespace := scope.execution_namespace) is not None
            and namespace.node is owner
        )
        if len(namespaces) != 1:
            raise ValueError("Class evidence requires one actual completed class scope")
        namespace = namespaces[0]
        return ClassNamespaceExecutionEvidence(
            frozenset(
                name
                for name, resolution in namespace.bindings.items()
                if resolution is not LexicalNameResolution.EXTERNAL
            ),
            tuple(
                site
                for site in self.sites
                if site.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
                and any(scope.class_declaration is owner for scope in site.scope_path)
            ),
        )


@dataclass(frozen=True)
class ClassNamespaceExecutionEvidence:
    """Final lexical bindings and execution obligations from one scope traversal."""

    binding_names: frozenset[str]
    sites: tuple[SourceEffectSite, ...]

    @property
    def effects(self) -> tuple[ClassNamespaceEffect, ...]:
        """Derive the inspection view without storing a second effect inventory."""
        return tuple(effect for site in self.sites for effect in site.effects)

    @classmethod
    def from_class(cls, owner: ast.ClassDef) -> ClassNamespaceExecutionEvidence:
        return SourceExecutionEffectEvidence.from_source(owner).class_evidence(owner)

    def require_closed(
        self,
        bindings: ModuleNominalBindingView,
        module: ParsedModule,
        owner: ast.ClassDef,
    ) -> None:
        bindings.native_reference_environment(module).require_class_creation(owner)


class _ClassNamespaceEffectProjection(ast.NodeVisitor):
    """Select one node's effects; the scope collector alone traverses children."""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_effect(DefinitionHeaderEffect, node)
        self._record_effect(ClassCreationEffect, node)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self._record_effect(ImportSourceEffect, node)

    visit_ImportFrom = visit_Import

    def visit_Attribute(self, node: ast.Attribute) -> None:
        effect_type = (
            ReferenceAccessEffect
            if isinstance(node.ctx, ast.Load)
            else NamespaceWriteEffect
        )
        self._record_effect(effect_type, node)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._record_effect(ExpressionStatementEffect, node)

    @classmethod
    def project(
        cls, trigger: ast.AST, scope: _DeclarationDependencyCollector
    ) -> tuple[ClassNamespaceEffect, ...]:
        """Classify this trigger only, retaining every invocation's obligations."""
        projection = cls(scope)
        projection.visit(trigger)
        return tuple(projection.effects)

    def __init__(self, scope: _DeclarationDependencyCollector) -> None:
        self.scope = scope
        self.effects: list[ClassNamespaceEffect] = []

    def _record_effect(
        self, effect_type: type[ClassNamespaceEffect], node: ast.AST
    ) -> None:
        self.effects.append(effect_type(node, self.scope.use))

    def generic_visit(self, node: ast.AST) -> None:
        # Unknown executable forms require proof rather than implicit trust.
        if isinstance(node, (ast.expr, ast.stmt)):
            self._record_effect(LiteralSourceEffect, node)

    def visit_Pass(self, node: ast.AST) -> None:
        pass

    def visit_Name(self, node: ast.Name) -> None:
        # Loads are captured by their consuming operation. Stores retain their
        # distinct mutation cut, after the right-hand-side capture.
        if not isinstance(node.ctx, ast.Load):
            self._record_effect(BindingSourceEffect, node)

    visit_Delete = visit_Lambda = visit_Tuple = visit_List = visit_Pass
    visit_Global = visit_Nonlocal = visit_Pass

    def visit_Call(self, node: ast.Call) -> None:
        self._record_effect(CallClassNamespaceEffect, node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        effect_type = (
            SubscriptionClassNamespaceEffect
            if isinstance(node.ctx, ast.Load)
            else ItemWriteEffect
        )
        self._record_effect(effect_type, node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self._record_effect(DictionaryConstructionEffect, node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._record_effect(DefinitionHeaderEffect, node)
        self._record_effect(BindingSourceEffect, node)
        for decorator in node.decorator_list:
            self._record_effect(DescriptorClassNamespaceEffect, decorator)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign | ast.AnnAssign | ast.NamedExpr) -> None:
        self._record_effect(AssignmentSourceEffect, node)

    visit_NamedExpr = visit_Assign

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit_Assign(node)
        if node.simple and self.scope.scopes[-1].records_variable_annotations:
            self._record_effect(AnnotationStorageEffect, node)

    def visit_If(self, node: ast.If | ast.IfExp) -> None:
        self._record_effect(LiteralSourceEffect, node.test)

    visit_IfExp = visit_If

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        # Python obtains the outer iterator immediately; its body is deferred.
        self._record_effect(LiteralIterationEffect, node.generators[0].iter)


class _SourceExecutionEffectCollector(_DeclarationDependencyCollector):
    """Attach obligations to the existing scope path without a second traversal."""

    def __init__(self) -> None:
        super().__init__()
        self.effect_sites: list[SourceEffectSite] = []
        self.completed_scopes: list[LexicalScopeABC] = []

    def visit(self, node: ast.AST) -> None:
        if self.scopes:
            effects = _ClassNamespaceEffectProjection.project(node, self)
            if effects:
                self.effect_sites.append(
                    SourceEffectSite(
                        node,
                        effects,
                        tuple(self.scopes),
                        self.binding_phase,
                    )
                )
        super().visit(node)

    @contextmanager
    def _scope(self, scope: LexicalScopeABC) -> Iterator[None]:
        with super()._scope(scope):
            yield
        self.completed_scopes.append(scope)
