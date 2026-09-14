"""Native subscription families own their argument-effect obligations."""

from __future__ import annotations

import ast
import builtins
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import (
    BuiltinFunctionType,
    GenericAlias,
)
from typing import (
    ClassVar,
    TYPE_CHECKING,
)

from .captured_reference import (
    CapturedReferenceResolution,
    CapturedReferenceViolation,
    NativeTypeCapture,
    OpaqueCapturedObjectOperations,
    CompletedSourceOperation,
)
from .descriptor_algebra import AliasProperty
from .native_compilation import (
    CPythonContainerConstruction,
    CPythonTypingConstruction,
    NativeCreationBackend,
)

from .native_declarations import (
    NativeDeclaration,
    NativeDeclarationFamily,
)
from .product_flow import (
    CompactFlowContext,
    CompactFlowValue,
    CompactSubscription,
)

NATIVE_BUILTIN_DECLARATIONS = tuple(
    NativeDeclaration(declaration)
    for declaration in vars(builtins).values()
    if isinstance(declaration, (type, BuiltinFunctionType))
)


if TYPE_CHECKING:
    from .native_reference import NativeReferenceEnvironment


@dataclass(frozen=True, eq=False)
class InertNativeArgumentWitness:
    """Binding representative derived from one original read, never target identity.

    The source owns the syntax and read. Inspection admits only native values
    with inert argument operations; it does not execute the source expression or
    establish its completion. The bound operation owns that separate prerequisite.
    """

    environment: NativeReferenceEnvironment
    read: CompactFlowValue

    @property
    def value(self) -> object:
        operation = self.environment.source.value_operation(self.read)
        return NativeArgumentInspection(self.environment).visit(operation.node)


class NativeSubscriptionAuthority(NativeDeclarationFamily, CompletedSourceOperation):
    """One native subscription consuming its original ordered source operands."""

    node = AliasProperty[ast.Subscript]("operation.node")
    invocation = AliasProperty[CompactSubscription]("operation.event")

    @cached_property
    def inspected_argument(self) -> InertNativeArgumentWitness:
        return InertNativeArgumentWitness(
            self.environment,
            CompactFlowValue(self.context, self.invocation.argument_use),
        )

    @classmethod
    def for_subscription(
        cls,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        invocation: CompactSubscription,
    ) -> NativeSubscriptionAuthority:
        operation = environment.source_operation(context, invocation)
        receiver = environment.kernel._read_use(
            invocation.receiver_use, context, frozenset()
        )
        return cls.select_from_capture(receiver)(environment, operation)

    def require_operation_kind(self) -> None:
        if not isinstance(self.operation.node, ast.Subscript) or not isinstance(
            self.operation.event, CompactSubscription
        ):
            raise TypeError("Subscription requires an actual load invocation")
        invocation = self.invocation
        if not any(item is invocation for item in self.context.flow.subscriptions):
            raise ValueError("Subscription is absent from its actual flow")
        if invocation.position.branch_path or invocation.position.evaluation_path:
            raise ValueError("Conditional or repeated subscription remains unproved")
        previous = None
        for node, receipt in (
            (self.node.value, invocation.receiver_use),
            (self.node.slice, invocation.argument_use),
        ):
            original = self.environment.source.value_operation(
                CompactFlowValue(self.context, receipt)
            )
            if original.node is not node:
                raise ValueError("Subscription operand is not its original production")
            if not receipt.position.dominates(invocation.position) or (
                previous is not None and not previous.dominates(receipt.position)
            ):
                raise ValueError(
                    "Subscription operands have no original ordered completion"
                )
            previous = receipt.position

    @cached_property
    def receiver(self) -> CapturedReferenceResolution:
        return self.environment.kernel._read_use(
            self.invocation.receiver_use, self.context, frozenset()
        )

    @cached_property
    def argument(self) -> CapturedReferenceResolution:
        return self.environment.kernel._read_use(
            self.invocation.argument_use, self.context, frozenset()
        )

    @cached_property
    def declaration(self) -> NativeDeclaration:
        return self.receiver.require_native(self.native_declarations)

    def require_closed(self) -> None:
        _ = self.completed

    @cached_property
    def completed(self) -> None:
        """Reuse successful construction proof within this original activation."""
        _ = self.declaration
        self.argument.require_closed()
        self.require_invocation()

    @abstractmethod
    def require_invocation(self) -> None:
        """Discharge this native protocol's binding and completion obligations."""
        raise NotImplementedError

    def require_inspected_result(self) -> object:
        """Construction alone does not prove native hashing or metadata access."""
        self.require_closed()
        raise ValueError("Subscription result hashing or metadata remains unproved")


class BuiltinGenericAliasSubscription(NativeSubscriptionAuthority):
    native_declarations = CPythonContainerConstruction.generic_alias_origins

    def require_invocation(self) -> None:
        NativeCreationBackend.current().require_generic_alias_construction(
            self.declaration
        )

    def result(self) -> CapturedReferenceResolution:
        self.require_closed()
        return BuiltinGenericAliasCapture(self)

    def require_inspected_result(self) -> object:
        """Return an inert binding representative, not the source alias object."""
        self.require_closed()
        return self.declaration.declaration[self.inspected_argument.value]


class ClassVariableSubscription(NativeSubscriptionAuthority):
    """ClassVar binding with explicit private-cache callback noninterference.

    A completing cache lookup can invoke resident keys' arbitrary equality or
    release callbacks. The supplied behavior condition excludes unmodelled
    mutations of admitted state, not the cache's own internal bookkeeping.
    Actual binding is independently checked; returned identity remains open.
    """

    native_declarations = (CPythonTypingConstruction.class_variable,)

    def require_invocation(self) -> None:
        NativeCreationBackend.current().require_classvar_binding(
            self.inspected_argument
        )
        self.environment.kernel.effects.require_native_behavior(self)


@dataclass(frozen=True, eq=False)
class BuiltinGenericAliasCapture(NativeTypeCapture, OpaqueCapturedObjectOperations):
    """An original native alias retaining operands, not an analyzer alias object."""

    authority: BuiltinGenericAliasSubscription
    native_type: ClassVar[type] = GenericAlias
    violation = CapturedReferenceViolation.UNPROVED_ACCESS
    origin = AliasProperty[CapturedReferenceResolution]("authority.receiver")
    argument = AliasProperty[CapturedReferenceResolution]("authority.argument")

    def require_closed(self) -> None:
        self.authority.require_closed()


class NativeArgumentInspection(ast.NodeVisitor):
    """Derive inert binding representatives without evaluating repository objects."""

    def __init__(self, environment: NativeReferenceEnvironment) -> None:
        self.environment = environment

    def generic_visit(self, node: ast.AST) -> object:
        raise ValueError("Native argument has unproved hashing or metadata effects")

    def visit_Name(self, node: ast.Name | ast.Attribute) -> object:
        return self.environment.require_native(
            node, NATIVE_BUILTIN_DECLARATIONS
        ).declaration

    visit_Attribute = visit_Name

    def visit_Constant(self, node: ast.Constant) -> object:
        return ast.literal_eval(node)

    def visit_Lambda(self, node: ast.Lambda) -> object:
        # Native function hashing/metadata and binding acceptance are inert.
        # This representative proves no source identity or metadata contents.
        # Original defaults/header execution remains a separate obligation.
        return lambda: None

    def visit_Tuple(self, node: ast.Tuple | ast.List) -> tuple[object, ...]:
        return tuple(self.visit(element) for element in node.elts)

    def visit_List(self, node: ast.List) -> list[object]:
        return list(self.visit_Tuple(node))

    def visit_Subscript(self, node: ast.Subscript) -> object:
        operation = self.environment.source.node_operation(node, CompactSubscription)
        context = self.environment.context_for_owner(operation.owner)
        return self.environment.subscription_authority(
            context, operation.event
        ).require_inspected_result()
