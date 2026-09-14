"""Original-source capture shared by native operation obligations."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import cast

from .captured_reference import CapturedReferenceKernel, CapturedReferenceResolution
from .descriptor_algebra import AliasProperty
from .native_call import (
    CallAuthority,
    NativeCallAuthority,
)
from .native_declarations import NativeDeclaration
from .native_subscription import NativeSubscriptionAuthority
from .product_flow import (
    CompactDefinitionTarget,
    CompactFlowContext,
    CompactFlowOwner,
    CompactFunctionCall,
    CompactMutation,
    CompactSubscription,
    SourceFlowOperation,
    SourceProductFlowProjection,
)


class NativeReferenceEnvironment(ABC):
    """One required original-source capture authority for native operations.

    The kernel's effect provider must admit the actual execution prefix. Equal
    AST coordinates, lexical spelling and source-qualified names are not native
    object identity. Consumers keep their separate operation obligations.
    """

    context_for_owner = AliasProperty[Callable[[CompactFlowOwner], CompactFlowContext]](
        "source.context_for_owner"
    )

    source_operation = AliasProperty[
        Callable[[CompactFlowContext, object], SourceFlowOperation]
    ]("source.source_operation")

    definition_operation = AliasProperty[Callable[[ast.AST], SourceFlowOperation]](
        "source.definition_operation"
    )

    def require_subscription(self, node: ast.Subscript) -> None:
        operation = self.source.node_operation(node, CompactSubscription)
        self.subscription_authority(
            self.context_for_owner(operation.owner), operation.event
        ).require_closed()

    def subscription_authority(
        self,
        context: CompactFlowContext,
        invocation: CompactSubscription,
    ) -> NativeSubscriptionAuthority:
        return NativeSubscriptionAuthority.for_subscription(self, context, invocation)

    def native_call_authority(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> NativeCallAuthority:
        return NativeCallAuthority.for_call(self, context, call)

    @cached_property
    def _call_authorities(self) -> dict[SourceFlowOperation, CallAuthority]:
        """Original invocations belong to this execution, not their source coordinates."""
        return {}

    def call_authority(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> CallAuthority:
        operation = self.source_operation(context, call)
        if operation not in self._call_authorities:
            self._call_authorities[operation] = self.kernel._read_use(
                call.target_use, context, frozenset()
            ).call_authority(self, context, call)
        return self._call_authorities[operation]

    @abstractmethod
    def require_import_operation(self, operation: SourceFlowOperation) -> None:
        """Require one actual import binding, without admitting later aliases."""
        raise NotImplementedError

    @abstractmethod
    def require_discard(self, node: ast.Expr) -> None:
        """Require evaluation and reference release at the actual result cut."""
        raise NotImplementedError

    @abstractmethod
    def require_call(self, node: ast.Call) -> None:
        """Require actual call execution, independently of returned identity."""
        raise NotImplementedError

    def definition_result(
        self,
        context: CompactFlowContext,
        binding: CompactMutation[CompactDefinitionTarget],
    ) -> CapturedReferenceResolution:
        """Query an actual definition result, not a newly fabricated source read.

        The original creation event is checked even when its installed name is
        rebound later. This query does not prove a later namespace slot stable.
        """
        self.source_operation(context, binding)
        if not isinstance(binding.target, CompactDefinitionTarget):
            raise ValueError("Definition result requires an actual definition binding")
        return self.kernel._definition_binding_resolution(
            context, binding.target.lexical_reference, binding, frozenset()
        )

    def capture_definition(self, node: ast.AST) -> CapturedReferenceResolution:
        """Resolve an actual definition result at its creation, not a later name read."""
        operation = self.definition_operation(node)
        return self.definition_result(
            self.context_for_owner(operation.owner),
            cast(CompactMutation[CompactDefinitionTarget], operation.event),
        )

    def capture_value(self, node: ast.expr) -> CapturedReferenceResolution:
        return self.kernel.read_source_value(self.source, node)

    @property
    @abstractmethod
    def source(self) -> SourceProductFlowProjection:
        raise NotImplementedError

    @property
    @abstractmethod
    def kernel(self) -> CapturedReferenceKernel:
        raise NotImplementedError

    @abstractmethod
    def require_namespace_write(self, node: ast.Attribute) -> None:
        """Require the actual write protocol separately from reading its receiver."""
        raise NotImplementedError

    def require_item_write(self, node: ast.Subscript | ast.AnnAssign) -> None:
        """Environments without source setter proof leave this operation open."""
        raise ValueError("Native item write remains unproved")

    def require_return(self, node: ast.Return) -> None:
        """Environments without function activation leave returned values open."""
        raise ValueError("Source return remains unproved")

    @abstractmethod
    def require_class_creation(self, node: ast.ClassDef) -> None:
        """Require class construction and installation, not merely value capture."""
        raise NotImplementedError

    @abstractmethod
    def require_import(self, node: ast.Import | ast.ImportFrom) -> None:
        """Require the actual import execution and binding protocol."""
        raise NotImplementedError

    @abstractmethod
    def require_binding_write(self, node: ast.AST) -> None:
        """Require storage and prior-value release at the original lexical mutation."""
        raise NotImplementedError

    def capture(self, node: ast.expr) -> CapturedReferenceResolution:
        return self.kernel.read_source(self.source, node)

    def require_native(
        self,
        node: ast.expr,
        declarations: tuple[NativeDeclaration, ...],
    ) -> NativeDeclaration:
        return self.capture(node).require_native(declarations)
