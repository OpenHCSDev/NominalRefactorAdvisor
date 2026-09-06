"""Positioned object capture under an explicitly admitted native environment.

This kernel does not prove arbitrary Python execution effects. A mandatory
effect authority must close the actual source prefix before any capture is
accepted. Source-origin names alone never authenticate a runtime object.
"""

from __future__ import annotations

import sys

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    field,
)
from enum import StrEnum
import inspect
from types import (
    MappingProxyType,
    ModuleType,
)
from typing import TypeAlias, cast

from .lexical_bindings import ImportedNameOrigin, ImportOriginResolverABC
from .native_declarations import NativeDeclaration
from .product_flow import (
    CompactAttributeTarget,
    CompactBindingResolverABC,
    CompactBindingVisit,
    CompactDefinitionTarget,
    CompactExactValueAlias,
    CompactFlowContext,
    CompactFlowPosition,
    CompactFlowRead,
    CompactFunctionTargetResolutionViolation,
    CompactItemTarget,
    CompactImportTarget,
    CompactMutation,
    CompactMutationResolverABC,
    CompactPositionedReference,
    CompactValueUse,
)
from .value_expression import LexicalValueReference


class CapturedReferenceViolation(StrEnum):
    UNPROVED_EFFECTS = "unproved_execution_effects"
    UNADMITTED_IMPORT = "unadmitted_native_import"
    UNPROVED_IMPORT_TRAVERSAL = "unproved_import_attribute_traversal"
    UNPROVED_BINDING = "unproved_binding"
    UNPROVED_ACCESS = "unproved_object_access"
    UNKNOWN_RECEIVER = "unknown_write_receiver"
    POSSIBLE_SLOT_WRITE = "possibly_preceding_slot_write"
    CYCLIC_BINDING = "cyclic_binding"


class CapturedReferenceResolution(ABC):
    @abstractmethod
    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        raise NotImplementedError

    @abstractmethod
    def write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        raise NotImplementedError

    @abstractmethod
    def require_native_identity(
        self, declaration: NativeDeclaration
    ) -> NativeDeclaration:
        """Require object identity, not stability of its mutable implementation."""
        raise NotImplementedError


@dataclass(frozen=True)
class OpenCapturedReference(CapturedReferenceResolution):
    violation: CapturedReferenceViolation
    mutation: CompactMutation | None = None

    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        return self

    def write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
        )

    def require_native_identity(
        self, declaration: NativeDeclaration
    ) -> NativeDeclaration:
        raise ValueError(f"Native object identity remains open: {self.violation.value}")


@dataclass(frozen=True, eq=False)
class CapturedNativeObject(CapturedReferenceResolution):
    """The actual initial object, never a source-qualified-name substitute."""

    value: object

    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        failure = resolver.initial.attribute_access_failure(self, attribute)
        if failure is not None:
            return failure
        return resolver._slot(self, attribute, context, position, pending)

    def write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        return resolver.initial.attribute_write_effect(self, query, mutation)

    def require_native_identity(
        self, declaration: NativeDeclaration
    ) -> NativeDeclaration:
        if self.value is not declaration.declaration:
            raise ValueError("Captured object is not the required native declaration")
        return declaration


class CapturedReferenceEffectsABC(ABC):
    """Required proof of the source execution surrounding a capture.

    Returning None asserts that every possibly preceding effect at this exact
    source/context/position is closed apart from direct slot identity effects
    checked by the kernel. This includes implicit operators, destruction,
    imports and import hooks, star imports, class construction, calls, and the
    initial frame globals/builtins. Missing compact records prove none of these.

    An admitted context must have native-island initial global/builtin lookup
    semantics for roots without a selected compact binding. Deferred scopes or
    custom frame environments require their own proof, not module fallback.
    The admitted native objects, captured initial sys.modules associations and
    native import behavior must remain valid throughout the query. A module's
    mutable display name is not an import handle. There is deliberately no
    permissive production implementation.
    """

    @abstractmethod
    def failure_for(
        self, context: CompactFlowContext, position: CompactFlowPosition
    ) -> OpenCapturedReference | None:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class InitialNativeIsland:
    """Actual module objects and their eagerly captured initial import handles.

    Only matching objects already in sys.modules acquire import handles.
    Capturing its filtered registry is evidence, not a display-name projection;
    analyzed imports are never run and later ambient changes cannot rewrite it.
    """

    modules: tuple[ModuleType, ...]
    builtin_module: ModuleType

    modules_by_name: Mapping[str, ModuleType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(type(module) is not ModuleType for module in self.modules):
            raise TypeError(
                "Initial native modules must have plain native module storage"
            )
        admitted_ids = {id(module) for module in self.modules}
        if len(admitted_ids) != len(self.modules):
            raise ValueError("Initial native module objects must be unique")
        if id(self.builtin_module) not in admitted_ids:
            raise ValueError(
                "The actual frame builtin module must belong to the island"
            )
        object.__setattr__(
            self,
            "modules_by_name",
            MappingProxyType(
                {
                    name: cast(ModuleType, module)
                    for name, module in tuple(sys.modules.items())
                    if type(name) is str and id(module) in admitted_ids
                }
            ),
        )

    def module(self, name: str | None) -> CapturedReferenceResolution:
        if name is None or name not in self.modules_by_name:
            return OpenCapturedReference(CapturedReferenceViolation.UNADMITTED_IMPORT)
        return CapturedNativeObject(self.modules_by_name[name])

    def imported_module(
        self, origin: ImportedNameOrigin
    ) -> CapturedReferenceResolution:
        bound_module = origin.qualified_name
        if (
            bound_module is None
            or origin.requested_module_name not in self.modules_by_name
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNADMITTED_IMPORT)
        # A dotted bound path requires Python's import attribute traversal/fallback.
        # Admission of a same-named module object alone cannot prove that traversal.
        if "." in bound_module:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_IMPORT_TRAVERSAL
            )
        return self.module(bound_module)

    @staticmethod
    def _has_module_data_descriptor(attribute: str) -> bool:
        try:
            descriptor = inspect.getattr_static(ModuleType, attribute)
        except AttributeError:
            return False
        return inspect.isdatadescriptor(descriptor)

    def attribute_access_failure(
        self, receiver: CapturedNativeObject, attribute: str
    ) -> OpenCapturedReference | None:
        if not any(
            receiver.value is module for module in self.modules
        ) or self._has_module_data_descriptor(attribute):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)
        return None

    def namespace_member(
        self, receiver: CapturedNativeObject, attribute: str
    ) -> CapturedReferenceResolution:
        """Read a raw namespace slot, including frame builtin dictionary lookup."""
        for module in self.modules:
            if receiver.value is module:
                try:
                    return CapturedNativeObject(vars(module)[attribute])
                except KeyError:
                    return OpenCapturedReference(
                        CapturedReferenceViolation.UNPROVED_ACCESS
                    )
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def attribute_write_effect(
        self,
        receiver: CapturedNativeObject,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        attribute = mutation.target.attribute_name
        if not any(
            receiver.value is module for module in self.modules
        ) or self._has_module_data_descriptor(attribute):
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, mutation
            )
        if receiver.value is query.receiver.value and attribute == query.attribute:
            return OpenCapturedReference(
                CapturedReferenceViolation.POSSIBLE_SLOT_WRITE, mutation
            )
        return None


@dataclass(frozen=True)
class CapturedSlotQuery:
    """A single object-slot obligation with its actual flow and recursion path."""

    receiver: CapturedNativeObject
    attribute: str
    context: CompactFlowContext
    pending: frozenset[CompactBindingVisit]


ImportQuery: TypeAlias = tuple[
    CompactFlowContext, CompactMutation, frozenset[CompactBindingVisit]
]


@dataclass(frozen=True)
class CapturedReferenceKernel(
    CompactBindingResolverABC[CapturedReferenceResolution],
    ImportOriginResolverABC[ImportQuery, CapturedReferenceResolution],
    CompactMutationResolverABC[CapturedSlotQuery, OpenCapturedReference | None],
):
    initial: InitialNativeIsland
    effects: CapturedReferenceEffectsABC

    def read(self, read: CompactFlowRead) -> CapturedReferenceResolution:
        return self._read_use(read.use, read.context, frozenset())

    def _read_use(
        self,
        use: CompactPositionedReference,
        context: CompactFlowContext,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        reference = use.lexical_reference
        if reference is None:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)
        return self._read_reference(reference, context, use.position, pending)

    def _read_reference(
        self,
        reference: LexicalValueReference,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        failure = self.effects.failure_for(context, position)
        if failure is not None:
            return failure
        root = LexicalValueReference(reference.root_name)
        binding = context.flow.binding_resolution_for(root.root_name, position)
        resolution = (
            self._slot(
                CapturedNativeObject(self.initial.builtin_module),
                root.root_name,
                context,
                position,
                pending,
            )
            if binding is None
            else binding.resolve_binding(self, context, root, position, pending)
        )
        for attribute in reference.attribute_path:
            resolution = resolution.access(self, attribute, context, position, pending)
        return resolution

    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        return self._read_use(alias.source_use, context, pending)

    def _installed_alias_resolution(
        self,
        resolution: CapturedReferenceResolution,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
    ) -> CapturedReferenceResolution:
        return resolution

    def _cyclic_binding_resolution(
        self,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.CYCLIC_BINDING)

    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNPROVED_BINDING, binding
        )

    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        origin = cast(CompactMutation[CompactImportTarget], binding).target.origin
        return origin.resolve(self, (context, binding, pending))

    def _module_import_resolution(
        self,
        origin: ImportedNameOrigin,
        context: ImportQuery,
    ) -> CapturedReferenceResolution:
        # Both the requested module and the object actually bound by native
        # import must be admitted; unaliased dotted imports bind the root.
        return self.initial.imported_module(origin)

    def _member_import_resolution(
        self,
        origin: ImportedNameOrigin,
        context: ImportQuery,
    ) -> CapturedReferenceResolution:
        flow_context, binding, pending = context
        return self.initial.module(origin.requested_module_name).access(
            self, origin.alias.name, flow_context, binding.position, pending
        )

    def _slot(
        self,
        receiver: CapturedNativeObject,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        failure = self.effects.failure_for(context, position)
        if failure is not None:
            return failure
        for mutation in context.flow.mutations:
            if not mutation.position.may_precede(position):
                continue
            visit = (context.owner_symbol, mutation)
            if visit in pending and mutation.target.bound_name is None:
                return OpenCapturedReference(
                    CapturedReferenceViolation.CYCLIC_BINDING, mutation
                )
            query = CapturedSlotQuery(receiver, attribute, context, pending | {visit})
            failure = mutation.resolve(self, query)
            if failure is not None:
                return failure
        return self.initial.namespace_member(receiver, attribute)

    def _binding_mutation_resolution(
        self,
        context: CapturedSlotQuery,
        mutation: CompactMutation,
        name: str,
    ) -> OpenCapturedReference | None:
        # The effect authority has separately admitted creation, import and
        # destruction effects. Lexical rebinding does not change object slots.
        return None

    def _receiver_mutation_resolution(
        self,
        context: CapturedSlotQuery,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> OpenCapturedReference:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
        )

    def _attribute_mutation_resolution(
        self,
        context: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        receiver = self._read_use(
            mutation.target.receiver_use, context.context, context.pending
        )
        return receiver.write_effect(self, context, mutation)

    def _item_mutation_resolution(
        self,
        context: CapturedSlotQuery,
        mutation: CompactMutation[CompactItemTarget],
    ) -> OpenCapturedReference:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNPROVED_EFFECTS, mutation
        )
