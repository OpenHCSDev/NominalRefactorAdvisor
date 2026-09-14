"""Positioned object capture under an explicitly admitted native environment.

This kernel does not prove arbitrary Python execution effects. A mandatory
effect authority must close the actual source prefix before any capture is
accepted. Source-origin names alone never authenticate a runtime object.
"""

from __future__ import annotations

import ast
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import (
    Iterator,
    Mapping,
)
from dataclasses import (
    InitVar,
    dataclass,
    field,
    replace,
)
from enum import StrEnum
from functools import cached_property, partial
from types import (
    MappingProxyType,
    ModuleType,
)
from typing import (
    ClassVar,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    cast,
)

from .class_mro import NativeMroBase
from .descriptor_algebra import AliasProperty

from .lexical_bindings import (
    ImportedNameOrigin,
    ImportOriginResolverABC,
    ImportFromModuleName,
)
from .native_compilation import (
    CPythonClassConstructionField,
    ExactNativeFunctionExecution,
    NativeBindingTransfer,
    NativeCreationBackend,
)
from .native_declarations import (
    NativeDeclaration,
    NativeScalar,
    NativeScalarValueABC,
)
from .product_flow import (
    CallResultValue,
    CompactAttributeTarget,
    CompactBindingResolverABC,
    CompactBindingSource,
    CompactBindingVisit,
    CompactDefinitionFlowOwner,
    CompactDefinitionResolverABC,
    CompactDefinitionSource,
    CompactDefinitionTarget,
    CompactExactValueAlias,
    CompactFlowContext,
    CompactFlowPosition,
    CompactFlowValue,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactFunctionTargetResolutionViolation,
    CompactImportTarget,
    CompactItemTarget,
    CompactMutation,
    CompactMutationResolverABC,
    CompactPositionedReference,
    CompactValueResolverABC,
    CompactValueUse,
    CompilerStoredValue,
    ForwardedResultValue,
    SourceFlowEvaluation,
    SourceFlowEvent,
    SourceFlowOperation,
    SourceProductFlowProjection,
)
from .value_expression import LexicalValueReference, TargetResolutionT

if TYPE_CHECKING:
    from .native_call import CallAuthority
    from .native_reference import NativeReferenceEnvironment
    from .source_entry import SourceModuleEntryPremise
    from .source_execution import (
        SourceDefinitionDecoratorApplicationABC,
        SourceFunctionActivationABC,
    )


class CapturedReferenceViolation(StrEnum):
    UNPROVED_EFFECTS = "unproved_execution_effects"
    UNADMITTED_IMPORT = "unadmitted_native_import"
    UNPROVED_IMPORT_TRAVERSAL = "unproved_import_attribute_traversal"
    UNPROVED_BINDING = "unproved_binding"
    UNPROVED_ACCESS = "unproved_object_access"
    UNKNOWN_RECEIVER = "unknown_write_receiver"
    POSSIBLE_SLOT_WRITE = "possibly_preceding_slot_write"
    CYCLIC_BINDING = "cyclic_binding"


class CapturedReferenceRejection(ValueError):
    """A fresh rejected query retaining its original capture-owned evidence."""

    violation = AliasProperty[CapturedReferenceViolation]("evidence.violation")
    evidence: OpenCapturedReference

    def __init__(self, evidence: OpenCapturedReference, query: str) -> None:
        super().__init__(f"{query} remains open: {evidence.violation.value}")
        self.evidence = evidence
        self.__cause__ = evidence.cause


class CapturedReferenceResolution(NativeScalarValueABC):

    def require_definition_application_argument(self) -> None:
        """Require this value as the actual input to a definition transformer."""
        self.require_closed()

    def definition_application_activation(
        self, application: SourceDefinitionDecoratorApplicationABC
    ) -> SourceFunctionActivationABC:
        """Select callable source activation from the actual application operand."""
        self.require_closed()
        raise ValueError("Definition application callable remains unproved")

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        """Require original source installation, not just a native object/type fact."""
        self.require_closed()
        raise ValueError("Source-bound native installation remains unproved")

    def require_fresh_function_namespace(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        """Require unchanged empty function custom storage at this actual source cut."""
        self.require_closed()
        raise ValueError("Fresh function storage at the source cut remains unproved")

    def require_nonabstract_member(self, prefix: AdmittedExecutionPrefixABC) -> None:
        """Require inert absence of ABCMeta's marker at this original source cut."""
        self.require_closed()
        raise ValueError("Abstract-member lookup at the source cut remains unproved")

    @property
    def native_type(self) -> type:
        """An unresolved capture does not establish an exact runtime type."""
        raise ValueError("Native value type remains unproved")

    def require_class_construction_field(
        self,
        field: CPythonClassConstructionField,
        namespace: NamespaceCreationEvidenceABC,
    ) -> None:
        """Discharge the value's native role at its actual construction destination."""
        field.require_default()

    def require_release_from(
        self,
        kernel: CapturedReferenceKernel,
        slot: CapturedSlotQuery,
    ) -> None:
        """Discharge release at the actual destination's admitted pre-write cut."""
        self.require_release_in(slot.prefix.endpoint.frame)

    def proves_same_object(self, other: CapturedReferenceResolution) -> bool:
        """Return a same-object proof; False leaves identity unknown, not distinct."""
        return False

    def call_authority(
        self,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CallAuthority:
        self.require_closed()
        raise ValueError("Callable execution remains unproved")

    @abstractmethod
    def require_class_installation(self) -> None:
        """Require this value's member creation hooks, separately from lookup."""
        raise NotImplementedError

    @abstractmethod
    def attribute_namespace(
        self, initial: InitialNativeIsland, attribute: str
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        """Require the actual storage for this native attribute, not vars alone."""
        raise NotImplementedError

    def source_definition(self) -> CompactDefinitionSource:
        """Require the canonical producer of this captured definition result.

        Closed lookup or native identity alone does not prove source provenance.
        The producer relation does not establish later member-slot identity.
        """
        self.require_closed()
        raise ValueError("Source definition identity remains unproved")

    def resolve_definition(
        self, resolver: CompactDefinitionResolverABC[TargetResolutionT]
    ) -> TargetResolutionT:
        context, binding = self.source_definition()
        return binding.kind.resolve_definition(
            resolver,
            f"{context.owner_symbol}.{binding.target.bound_name}",
            binding,
        )

    def require_definition_identity(self, owner: CompactDefinitionFlowOwner) -> None:
        """Compare against the producer proved by this result's own protocol."""
        _, binding = self.source_definition()
        if binding.target.owner is not owner:
            raise ValueError("Captured result belongs to another source definition")

    def require_release_in(self, frame: InitialNativeFrame) -> None:
        """An admitted active frame may supply an independent retained reference."""
        self.require_release()

    @abstractmethod
    def object_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        """Require an object's native vars protocol; a dictionary is not an object namespace."""
        raise NotImplementedError

    @abstractmethod
    def dictionary_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        """Require exact dictionary storage, not merely a module's namespace."""
        raise NotImplementedError

    @abstractmethod
    def as_builtin_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        """Resolve the module-or-dictionary protocol of a captured frame builtin value."""
        raise NotImplementedError

    @abstractmethod
    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        """Require native type-based construction without inherited creation callbacks."""
        raise NotImplementedError

    @abstractmethod
    def require_attribute_write(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        """Require ordinary attribute storage and old-value release at its actual cut."""
        raise NotImplementedError

    def require_item_write(
        self,
        resolver: CapturedReferenceKernel,
        key: NativeScalar,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        """Require dictionary storage plus prior-value and receiver release."""
        namespace = self.dictionary_namespace(resolver.initial)
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
            return
        prefix = resolver._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
            return
        namespace.require_available(resolver, prefix)
        NativeCreationBackend.current().require_dictionary_scalar_store(key)
        namespace.require_slot_release(resolver, key, context, position)
        # STORE_SUBSCR releases the consumed receiver as well as the key/value
        # operands. The newly retained RHS cannot prove receiver retention.
        if not prefix.endpoint.frame.retains_namespace(namespace):
            CapturedSlotQuery(
                namespace, key, prefix, frozenset()
            ).require_independent_reference(resolver, self)

    @abstractmethod
    def require_release(self) -> None:
        """Require decrement/destruction safety, not analyzer-held object liveness."""
        raise NotImplementedError

    @abstractmethod
    def require_closed(self) -> None:
        """Require a closed capture, without authenticating a particular object.

        The admitted prefix and lookup operation must both be closed. This
        does not prove calling, hashing, subscribing, or installing the result
        is inert; those consumers retain their own operation obligations.
        """
        raise NotImplementedError

    @abstractmethod
    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        occurrence: ContextualMutation,
        key: CapturedReferenceResolution,
    ) -> CapturedReferenceResolution | None:
        raise NotImplementedError

    @abstractmethod
    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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
    def require_native(
        self, declarations: tuple[NativeDeclaration, ...]
    ) -> NativeDeclaration:
        """Select an admitted declaration by actual captured object identity."""
        raise NotImplementedError

    def require_native_identity(
        self, declaration: NativeDeclaration
    ) -> NativeDeclaration:
        """Require one object's identity, not mutable implementation stability."""
        return self.require_native((declaration,))


class InstanceProtocolEvidenceABC(CapturedReferenceResolution):
    """Prove a requested native instance protocol on the actual captured class."""

    @abstractmethod
    def require_inert_instance_hooks(
        self,
        hooks: tuple[NativeDeclaration, ...],
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        raise NotImplementedError


class NativeTypeCapture(CapturedReferenceResolution, ABC):
    """A captured value with proven exact native type, independently of identity."""

    def _native_constant_value(self) -> object:
        self.require_closed()
        return self._native_scalar_value()

    def require_native_scalar(self) -> NativeScalar:
        self.require_closed()
        return super().require_native_scalar()

    def require_release(self) -> None:
        self.require_closed()
        NativeCreationBackend.current().require_inert_instance_release(self.native_type)

    @property
    @abstractmethod
    def native_type(self) -> type:
        raise NotImplementedError

    def require_class_installation(self) -> None:
        self.require_closed()
        NativeCreationBackend.current().require_inert_class_member_type(
            self.native_type
        )

    def require_nonabstract_member(self, prefix: AdmittedExecutionPrefixABC) -> None:
        self.require_closed()
        NativeCreationBackend.current().require_nonabstract_member(self.native_type)


class OpaqueCapturedObjectOperations(CapturedReferenceResolution, ABC):
    """An explicit captured kind can leave particular object protocols unknown."""

    violation: CapturedReferenceViolation

    def require_class_installation(self) -> None:
        raise self.rejection("Class member installation")

    def attribute_namespace(
        self, initial: InitialNativeIsland, attribute: str
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.open_reference()

    def rejection(self, operation: str) -> CapturedReferenceRejection:
        """Format the operation label; this text never selects proof behavior."""
        return CapturedReferenceRejection(self.open_reference(), operation)

    def open_reference(self) -> OpenCapturedReference:
        """Project an unproved operation without inventing a captured identity."""
        return OpenCapturedReference(self.violation)

    def as_builtin_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.open_reference()

    def dictionary_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.open_reference()

    def object_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.open_reference()

    def require_native(
        self, declarations: tuple[NativeDeclaration, ...]
    ) -> NativeDeclaration:
        raise self.rejection("Native object identity")

    def require_release(self) -> None:
        raise self.rejection("Object destruction")

    def require_attribute_write(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        raise self.rejection("Namespace write")

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        raise self.rejection("Class base")

    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return self.open_reference()

    def write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
        )

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        occurrence: ContextualMutation,
        key: CapturedReferenceResolution,
    ) -> CapturedReferenceResolution | None:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, occurrence.mutation
        )


@dataclass(frozen=True)
class OpenCapturedReference(OpaqueCapturedObjectOperations):
    violation: CapturedReferenceViolation
    mutation: CompactMutation | None = None
    cause: ValueError | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Keep original diagnostic links, but no captured execution stacks."""
        pending: list[BaseException] = [] if self.cause is None else [self.cause]
        visited: set[int] = set()
        while pending:
            error = pending.pop()
            if id(error) in visited:
                continue
            visited.add(id(error))
            BaseException.with_traceback(error, None)
            pending.extend(
                linked
                for linked in (error.__cause__, error.__context__)
                if linked is not None
            )
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)

    def require_closed(self) -> None:
        raise self.rejection("Object capture")

    def open_reference(self) -> OpenCapturedReference:
        return self


@dataclass(frozen=True, eq=False)
class NativeTypePremise(NativeTypeCapture, OpaqueCapturedObjectOperations):
    """Supplied exact-type evidence, without a target object identity."""

    declared_type: type
    native_type = AliasProperty[type]("declared_type")
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    def require_closed(self) -> None:
        pass


@dataclass(frozen=True, eq=False)
class CapturedNativeObject(NativeTypeCapture):
    """The actual initial object, never a source-qualified-name substitute."""

    value: object

    def _native_scalar_value(self) -> object:
        return self.value

    def proves_same_object(self, other: CapturedReferenceResolution) -> bool:
        """Compare actual captured objects, never equality or declaration spelling."""
        if not isinstance(other, CapturedNativeObject):
            return False
        self.require_closed()
        other.require_closed()
        return self.value is other.value

    def call_authority(
        self,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CallAuthority:
        self.require_closed()
        return environment.native_call_authority(context, call)

    @property
    def native_type(self) -> type:
        return type(self.value)

    def attribute_namespace(
        self, initial: InitialNativeIsland, attribute: str
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        return initial.attribute_namespace(self, attribute)

    def require_release_in(self, frame: InitialNativeFrame) -> None:
        if not frame.retains_initial_storage(self.value):
            self.require_release()

    def object_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        for module in initial.modules:
            if self.value is module:
                return initial.namespace_for_storage(vars(module))
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def dictionary_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        for namespace in initial.namespaces:
            if namespace.storage is self.value:
                return namespace
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def as_builtin_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        namespace = self.object_namespace(initial)
        return (
            self.dictionary_namespace(initial)
            if isinstance(namespace, OpenCapturedReference)
            else namespace
        )

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        """Admit the native root under the original capture and execution premise."""
        root = NativeMroBase.OBJECT.python_type
        self.require_native_identity(NativeDeclaration(root))
        prefix = resolver._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        NativeCreationBackend.current().require_static_type_release(root)
        return root

    def require_attribute_write(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        namespace = self.attribute_namespace(resolver.initial, attribute)
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
            return
        namespace.require_slot_release(resolver, attribute, context, position)

    def require_release(self) -> None:
        self.require_closed()
        NativeCreationBackend.current().require_object_release(self.value)

    def require_native(
        self, declarations: tuple[NativeDeclaration, ...]
    ) -> NativeDeclaration:
        matches = {
            id(declaration.declaration): declaration
            for declaration in declarations
            if self.value is declaration.declaration
        }
        if len(matches) != 1:
            raise ValueError("Captured object is not the required native declaration")
        return next(iter(matches.values()))

    def require_closed(self) -> None:
        # This leaf retains the actual object reached by an admitted lookup.
        pass

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        occurrence: ContextualMutation,
        key: CapturedReferenceResolution,
    ) -> CapturedReferenceResolution | None:
        return resolver.initial.item_write_effect(
            resolver, self, query, occurrence, key
        )

    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        namespace = self.attribute_namespace(resolver.initial, attribute)
        if isinstance(namespace, OpenCapturedReference):
            return namespace
        result = resolver._slot(namespace, attribute, context, position, pending)
        return (
            OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)
            if result is None
            else result
        )

    def write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        namespace = self.attribute_namespace(
            resolver.initial, mutation.target.attribute_name
        )
        if isinstance(namespace, OpenCapturedReference):
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, mutation
            )
        return query.write_effect(namespace, mutation.target.attribute_name, mutation)


NamespaceEntryT = TypeVar("NamespaceEntryT")


class NamespaceEvidenceABC(ABC):
    """A canonical admitted namespace identity and its initial binding evidence.

    Source-created namespaces must carry proven creation, not substitute a
    dictionary made by the analyzer. Within one admission aliases share the
    canonical owner; raw dictionary equality never authenticates namespace
    identity. None from member is proved initial absence, not uncertainty.
    """

    def require_native_creator(self, execution: ExactNativeFunctionExecution) -> None:
        """Require the original raw code whose active locals this namespace owns."""
        raise ValueError("Native creator activation remains unproved")

    def require_available(
        self, kernel: CapturedReferenceKernel, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        raise ValueError(
            "Namespace availability at this execution cut remains unproved"
        )

    @property
    def initial_names(self) -> frozenset[NativeScalar]:
        raise ValueError("Complete initial namespace membership remains unproved")

    def captured_dictionary(self) -> CapturedReferenceResolution:
        """A namespace is not automatically an observed dictionary value."""
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def require_slot_release(
        self,
        resolver: CapturedReferenceKernel,
        key: NativeScalar,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> CapturedReferenceResolution | None:
        """Resolve destination storage and release under its admitted active frame."""
        prefix = resolver._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        prefix = cast(AdmittedExecutionPrefixABC, prefix)
        previous = resolver._namespace_resolution(self, key, prefix, frozenset())
        if previous is not None:
            previous.require_release_from(
                resolver, CapturedSlotQuery(self, key, prefix, frozenset())
            )
        return previous

    @staticmethod
    def capture_initial_entries(
        entries: dict[NativeScalar, NamespaceEntryT],
    ) -> Mapping[NativeScalar, NamespaceEntryT]:
        if type(entries) is not dict:
            raise TypeError("A native namespace requires exact dictionary storage")
        for key in entries:
            NamespaceEvidenceABC.require_key(key)
        return MappingProxyType(entries.copy())

    @staticmethod
    def require_key(key: NativeScalar) -> None:
        if not NativeScalarValueABC.supports_scalar(key):
            raise TypeError("Native dictionary lookup requires an exact scalar key")

    def member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        self.require_key(key)
        return self._member(key)

    @abstractmethod
    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        raise NotImplementedError

    @abstractmethod
    def require_admitted(self, initial: InitialNativeIsland) -> None:
        """Require this exact owner in the specified initial execution premise."""
        raise NotImplementedError

    @abstractmethod
    def is_initial_storage(self, storage: object) -> bool:
        """Compare observed initial storage, never infer from source spelling.

        A source-created leaf may prove disjointness only from its admitted
        fresh-after-initial creation relation, not from representation type.
        """
        raise NotImplementedError


class RecordedNamespace(NamespaceEvidenceABC, ABC):
    """Complete initial keys derive from the existing admitted entry record."""

    initial_entries: Mapping[NativeScalar, object]

    @cached_property
    def initial_names(self) -> frozenset[NativeScalar]:
        return frozenset(self.initial_entries)


class NamespaceCreationEvidenceABC(NamespaceEvidenceABC):
    """Required provenance for a source-generated fresh namespace.

    Implementations must establish actual admitted creation and uniqueness of
    its activation under an explicit analyzed-program entry premise. A source
    span/qualname or newly allocated analyzer dict is not creation evidence.
    The source execution policy owns concrete entry and class creation proofs.
    """

    def require_available(
        self, kernel: CapturedReferenceKernel, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        self.require_admitted(kernel.initial)
        if not prefix.endpoint.frame.retains_namespace(self):
            raise ValueError("Source namespace is not retained by this active frame")

    @property
    @abstractmethod
    def initial(self) -> InitialNativeIsland:
        raise NotImplementedError

    def is_initial_storage(self, storage: object) -> bool:
        return False


class CreatedNamespaceDictionary(
    NamespaceCreationEvidenceABC,
    NativeTypeCapture,
    OpaqueCapturedObjectOperations,
):
    """An admitted fresh dictionary keeps its namespace and value identity together."""

    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    native_type: ClassVar[type] = dict

    def require_release_from(
        self, kernel: CapturedReferenceKernel, slot: CapturedSlotQuery
    ) -> None:
        """An exact dictionary releases its currently proved contents, not its birth state."""
        self.require_admitted(kernel.initial)
        slot.prefix.require_admitted(kernel.initial)
        if slot.prefix.endpoint.frame.retains_namespace(self):
            return
        contents = NamespaceMemberInventory(kernel, self, slot.prefix)
        for key in contents.names:
            contents.require_member(key).require_release()

    def resolve_definition(
        self, resolver: CompactDefinitionResolverABC[TargetResolutionT]
    ) -> TargetResolutionT:
        self.require_closed()
        return resolver._non_definition_resolution()

    def require_release_in(self, frame: InitialNativeFrame) -> None:
        self.require_admitted(self.initial)
        if not frame.retains_namespace(self):
            self.require_release()

    def require_closed(self) -> None:
        self.require_admitted(self.initial)

    def proves_same_object(self, other: CapturedReferenceResolution) -> bool:
        self.require_closed()
        return other is self

    def captured_dictionary(self) -> CapturedReferenceResolution:
        self.require_closed()
        return self

    def dictionary_namespace(
        self, initial: InitialNativeIsland
    ) -> NamespaceEvidenceABC:
        self.require_admitted(initial)
        return self

    as_builtin_namespace = dictionary_namespace

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        occurrence: ContextualMutation,
        key: CapturedReferenceResolution,
    ) -> CapturedReferenceResolution | None:
        self.require_admitted(resolver.initial)
        return (
            query.item_key_write_effect(resolver, key, occurrence)
            if query.namespace is self
            else None
        )


class EmptyDictionaryCreation(CreatedNamespaceDictionary):
    """Proved empty birth storage; subsequent writes use the shared namespace kernel."""

    @property
    def initial_names(self) -> frozenset[NativeScalar]:
        self.require_admitted(self.initial)
        return frozenset()

    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        self.require_admitted(self.initial)
        return None


@dataclass(frozen=True, eq=False)
class NativeNamespace(RecordedNamespace):
    """One observed native dictionary and its admitted initial scalar-key state.

    Key admission precedes lookup/copy: even an exact dict can contain foreign
    keys whose equality runs code. The effect proof must preserve this invariant
    and account for later index, destruction and external mutation effects.
    """

    storage: dict[NativeScalar, object]
    initial_entries: Mapping[NativeScalar, object] = field(init=False, repr=False)

    def require_available(
        self, kernel: CapturedReferenceKernel, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        self.require_admitted(kernel.initial)

    def captured_dictionary(self) -> CapturedReferenceResolution:
        return CapturedNativeObject(self.storage)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_entries", self.capture_initial_entries(self.storage)
        )

    def _member(self, key: NativeScalar) -> CapturedNativeObject | None:
        if key not in self.initial_entries:
            return None
        return CapturedNativeObject(self.initial_entries[key])

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if not any(self is namespace for namespace in initial.namespaces):
            raise ValueError("Frame namespace owner belongs to a different admission")

    def is_initial_storage(self, storage: object) -> bool:
        return self.storage is storage


@dataclass(frozen=True, eq=False)
class InitialNativeFrame:
    """Actual admitted namespaces, not globals' mutable __builtins__ spelling.

    The source/position admission must prove these native lookup and write
    destinations apply to the current activation. Source parentage alone is
    insufficient, including compiler-created generic frames. A namespace
    component can remain explicitly open until a lookup demands it; known
    globals do not require an unused builtin fallback to be resolved. Nonlocal binding
    requires a closure relation not represented by these namespace handles.
    """

    locals: NamespaceEvidenceABC | OpenCapturedReference
    globals: NamespaceEvidenceABC | OpenCapturedReference
    builtins: NamespaceEvidenceABC | OpenCapturedReference

    def retains_namespace(self, namespace: NamespaceEvidenceABC) -> bool:
        """Active frame slots retain this exact namespace independently of its contents."""
        return any(value is namespace for value in self.namespace_values)

    def retains_initial_storage(self, value: object) -> bool:
        """Actual active frame slots retain their namespace dictionaries, not arbitrary entries."""
        return any(
            namespace.is_initial_storage(value)
            for namespace in self.namespace_values
            if isinstance(namespace, NamespaceEvidenceABC)
        )

    @property
    def namespace_values(
        self,
    ) -> tuple[NamespaceEvidenceABC | OpenCapturedReference, ...]:
        return (self.locals, self.globals, self.builtins)

    def binding_namespace(
        self,
        context: CompactFlowContext,
        name: str,
    ) -> NamespaceEvidenceABC | OpenCapturedReference:
        if name in context.flow.nonlocal_binding_names:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        if name in context.flow.global_binding_names:
            return self.globals
        return self.locals

    def initial_lookup_namespaces(
        self,
        context: CompactFlowContext,
        name: str,
    ) -> (
        tuple[NamespaceEvidenceABC | OpenCapturedReference, ...] | OpenCapturedReference
    ):
        binding_namespace = self.binding_namespace(context, name)
        if isinstance(binding_namespace, OpenCapturedReference):
            return binding_namespace
        if context.flow.local_binding_hides_outer_lookup(name):
            return (binding_namespace,)
        return tuple(
            {
                id(namespace): namespace
                for namespace in (binding_namespace, self.globals, self.builtins)
            }.values()
        )


class SourceActivationAuthorityABC(ABC):
    """One original operation at its declaration-owned activation cut.

    Ordinary invocations and source definitions retain their existing owners,
    fields and prefixes. This contract authenticates their shared source entry;
    it does not close a class body or establish a native result.
    """

    environment: NativeReferenceEnvironment
    operation: SourceFlowOperation

    @property
    @abstractmethod
    def activation_context(self) -> CompactFlowContext:
        raise NotImplementedError

    @property
    @abstractmethod
    def activation_position(self) -> CompactFlowPosition:
        raise NotImplementedError

    @property
    @abstractmethod
    def activation_prefix(self) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError

    @abstractmethod
    def require_operation_kind(self) -> None:
        """Require this owner's original node, event and flow membership."""
        raise NotImplementedError

    def require_original_operation(self) -> None:
        original = self.environment.source_operation(
            self.activation_context, self.operation.event
        )
        if original is not self.operation:
            raise ValueError(
                "Operation requires its original canonical source operation"
            )
        self.require_operation_kind()

    def require_condition_activation(self, entry: SourceModuleEntryPremise) -> None:
        """Authenticate the original activation without substituting another cut."""
        if (
            self.environment.source is not entry.source
            or self.environment.kernel.initial is not entry.initial
        ):
            raise ValueError(
                "Operation condition belongs to a different source activation"
            )
        self.require_original_operation()
        endpoint = self.activation_prefix.endpoint
        if (
            self.activation_context is not endpoint.context
            or endpoint.context
            is not entry.source.context_for_owner(self.operation.owner)
            or endpoint.position != self.activation_position
            or endpoint.frame.globals is not entry
        ):
            raise ValueError("Operation condition requires its original activation cut")


@dataclass(frozen=True, eq=False)
class SourceOperationAuthority(SourceActivationAuthorityABC):
    """One original operation and its admitted execution prefix.

    Operation families supply only their node/event obligations; the canonical
    source join, activation context and prefix remain owned here.
    """

    environment: NativeReferenceEnvironment
    operation: SourceFlowOperation

    activation_context = AliasProperty[CompactFlowContext]("context")

    activation_position = AliasProperty[CompactFlowPosition]("operation.position")

    activation_prefix = AliasProperty["AdmittedExecutionPrefixABC"]("prefix")

    def __post_init__(self) -> None:
        self.require_original_operation()
        _ = self.prefix

    @cached_property
    def context(self) -> CompactFlowContext:
        return self.environment.context_for_owner(self.operation.owner)

    @cached_property
    def prefix(self) -> AdmittedExecutionPrefixABC:
        prefix = self.environment.kernel._admitted_prefix(
            self.context, self.operation.position
        )
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        return cast(AdmittedExecutionPrefixABC, prefix)


class CompletedSourceOperation(
    OpaqueCapturedObjectOperations, SourceOperationAuthority
):
    """A completed invocation retains a result without knowing its object protocols.

    The original operation owns this evidence. Completion supplies neither an
    exact type nor same-object identity, hashing, member hooks or safe release.
    """

    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    def result(self) -> CapturedReferenceResolution:
        self.require_original_operation()
        self.require_closed()
        return self

    @cached_property
    def captured_result(self) -> CapturedReferenceResolution:
        """Publish only a closed result under its original invocation owner."""
        self.require_original_operation()
        result = self.result()
        result.require_closed()
        return result


class AdmittedExecutionPrefixABC(ABC):
    """A supplied activation/order proof, not an inferred source-parent relation.

    Intervals retain their own context, canonical activation frame and local
    cuts. Composition declares execution order; positions are compared only
    inside one flow. The effect authority closes omitted Python effects before
    supplying this relation. Bindings and writes remain derived from each flow.
    """

    @abstractmethod
    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        """Contribute this receipt's actual ordered parts or atomic interval."""
        raise NotImplementedError

    @property
    def intervals(self) -> tuple[SingleFlowPrefix, ...]:
        """Project ordered occurrences without recursing through prior prefixes."""
        pending: list[AdmittedExecutionPrefixABC] = [self]
        intervals: list[SingleFlowPrefix] = []
        while pending:
            pending.pop()._expand_intervals(pending, intervals)
        return tuple(intervals)

    @property
    def endpoint(self) -> SingleFlowPrefix:
        """Follow the declared traversal's last child, without expanding prior history."""
        pending: list[AdmittedExecutionPrefixABC] = [self]
        terminal: list[SingleFlowPrefix] = []
        while not terminal:
            # Expansion supplies a reversed stack for chronological traversal.
            # Its first entry is therefore the last chronological child.
            selected = pending[0]
            pending.clear()
            selected._expand_intervals(pending, terminal)
        return terminal[-1]

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        previous: dict[tuple[int, int], SingleFlowPrefix] = {}
        for interval in self.intervals:
            initial.require_frame(interval.frame)
            activation = (id(interval.context), id(interval.frame))
            if activation in previous:
                earlier = previous[activation]
                if earlier.position is None or interval.after != earlier.position:
                    raise ValueError(
                        "Execution intervals must join the same activation cut"
                    )
            elif interval.after is not None:
                raise ValueError(
                    "A prefix cannot omit its activation's earlier interval"
                )
            previous[activation] = interval

    def require_event(
        self,
        context: CompactFlowContext,
        event: SourceFlowEvent,
        frame: InitialNativeFrame,
    ) -> SingleFlowPrefix:
        """Require one original event occurrence in its actual activation frame."""
        if context.flow.graph_nodes_by_identity.get(id(event)) is not event:
            raise ValueError("Execution event is absent from its original flow")
        occurrences = tuple(
            interval
            for interval in self.intervals
            if interval.context is context
            and interval.frame is frame
            and interval.contains(event)
        )
        if len(occurrences) != 1:
            raise ValueError("Execution event has no unique occurrence at this cut")
        interval = occurrences[0]
        interval.require_definite(event)
        return interval

    def entry_contents(
        self, kernel: CapturedReferenceKernel, namespace: NamespaceEvidenceABC
    ) -> NamespaceContentsABC:
        """Entry work belongs to the namespace's original local-frame owner.

        Initial native namespaces and non-frame storage retain their supplied
        namespace evidence; they do not acquire a source activation owner.
        """
        for interval in reversed(self.intervals):
            if interval.frame.locals is namespace:
                kernel = interval.capture_kernel(kernel)
                break
        return kernel.effects.entry_contents(kernel, namespace, self)

    def binding_sources(
        self, namespace: NamespaceEvidenceABC, name: NativeScalar
    ) -> Iterator[
        tuple[SingleFlowPrefix, CompactBindingSource] | OpenCapturedReference
    ]:
        """Resolve selected writes at their actual execution interval, not a later cut.

        A resumed parent's selected binding can precede a completed child's global
        write. Its unchanged source-flow selection therefore contributes only at
        the interval containing that installation. Unresolved selection stays open.
        """
        # Only Unicode keys can also be installed by lexical STORE_* events.
        # Non-text dictionary slots still replay their original item mutations.
        if type(name) is not str:
            return
        for interval in reversed(self.intervals):
            destination = interval.frame.binding_namespace(interval.context, name)
            if isinstance(destination, OpenCapturedReference):
                yield destination
                continue
            if destination is not namespace:
                continue
            binding = interval.context.flow.stored_binding_resolution_for(
                name, interval.position
            )
            if binding is None:
                continue
            if binding.mutation is not None and not interval.contains(binding.mutation):
                continue
            yield interval, binding

    def mutation_occurrences(self) -> Iterator[ContextualMutation]:
        for interval in self.intervals:
            yield from interval.mutation_occurrences()


@dataclass(frozen=True, eq=False)
class SingleFlowPrefix(AdmittedExecutionPrefixABC):
    """One admitted interval; None end explicitly means completed execution.

    A non-None start is a continuation cut, only valid when composition retains
    its preceding interval. Completion is supplied evidence, never a fabricated
    last event number. Repeated activations require distinct canonical frames.
    """

    context: CompactFlowContext
    frame: InitialNativeFrame
    position: CompactFlowPosition | None
    after: CompactFlowPosition | None = None

    def capture_kernel(
        self, observer: CapturedReferenceKernel
    ) -> CapturedReferenceKernel:
        selected = self._capture_kernel(observer)
        if selected.initial is not observer.initial:
            raise ValueError("Interval capture belongs to a different native admission")
        selected.require_interval_frame(self)
        return selected

    def _capture_kernel(
        self, observer: CapturedReferenceKernel
    ) -> CapturedReferenceKernel:
        """An explicitly supplied interval must match the supplied observer's frame."""
        return observer

    def may_overlap_positions(
        self,
        entry: CompactFlowPosition,
        exit: CompactFlowPosition,
    ) -> bool:
        """Retain uncertain ordering and repeated cuts under one interval law."""
        if self.after is not None and self.after == self.position:
            return False
        return (self.position is None or entry.may_precede_cut(self.position)) and (
            self.after is None or self.after.may_precede_cut(exit)
        )

    def may_overlap_evaluation(self, evaluation: SourceFlowEvaluation) -> bool:
        """Retain possible observer overlap under existing strict, loop-aware cuts."""
        return self.may_overlap_positions(evaluation.entry, evaluation.exit)

    def require_definite(self, event: SourceFlowEvent) -> None:
        """Require an actual preceding event, not a possibly executed branch write."""
        if not self.contains(event):
            raise ValueError("Event is outside the admitted execution interval")
        if self.position is None:
            if event.position.branch_path or event.position.evaluation_path:
                raise ValueError("Conditional execution membership remains unproved")
        elif not event.position.dominates(self.position):
            raise ValueError("Definite execution membership remains unproved")

    @cached_property
    def _mutation_occurrences(self) -> tuple[ContextualMutation, ...]:
        """Project actual writes at this immutable interval's own cuts once."""
        return tuple(
            ContextualMutation(self, mutation)
            for mutation in self.context.flow.mutations
            if self.contains(mutation)
        )

    def __post_init__(self) -> None:
        if (
            self.after is not None
            and self.position is not None
            and self.after != self.position
            and not self.after.dominates(self.position)
        ):
            raise ValueError("Flow interval requires proved ordered cuts")

    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        intervals.append(self)

    def mutation_occurrences(self) -> Iterator[ContextualMutation]:
        return iter(self._mutation_occurrences)

    def contains(self, mutation: SourceFlowEvent) -> bool:
        """Half-open [after, position), retaining uncertain preceding events."""
        if self.after is not None and self.after == self.position:
            return False
        return (
            self.position is None
            or (
                mutation.position != self.position
                and mutation.position.may_precede(self.position)
            )
        ) and (self.after is None or self.after.may_precede(mutation.position))


@dataclass(frozen=True, eq=False)
class CapturedFlowPrefix(SingleFlowPrefix):
    """A source interval retaining the kernel of its original activation."""

    kernel: CapturedReferenceKernel = field(kw_only=True)

    def _capture_kernel(
        self, observer: CapturedReferenceKernel
    ) -> CapturedReferenceKernel:
        return self.kernel


@dataclass(frozen=True, eq=False)
class SequentialExecutionPrefix(AdmittedExecutionPrefixABC):
    """Admitted execution parts, retaining each flow's own positioned cuts."""

    parts: tuple[AdmittedExecutionPrefixABC, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("An execution prefix requires at least one part")

    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        pending.extend(reversed(self.parts))


@dataclass(frozen=True, eq=False)
class ChildExecutionPrefix(AdmittedExecutionPrefixABC):
    """A supplied child-entry relation retaining the actual definition event.

    Owner linkage validates the receipt, not Python activation. In particular,
    the parent's supplied cut is not inferred from header_position, and neither
    the builder lookup nor the child builtins capture is inferred from parentage.
    """

    parent: AdmittedExecutionPrefixABC
    definition: CompactMutation[CompactDefinitionTarget]
    child: AdmittedExecutionPrefixABC

    def __post_init__(self) -> None:
        parent = self.parent.endpoint
        child = self.child.intervals[0]
        if not any(self.definition is event for event in parent.context.flow.mutations):
            raise ValueError("Child entry requires its actual parent definition event")
        if self.definition.target.owner is not child.context.flow.owner:
            raise ValueError("Child entry requires the definition's actual flow owner")
        if parent.position is None or child.after is not None:
            raise ValueError(
                "Child entry requires a parent cut and a fresh child interval"
            )

    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        pending.extend((self.child, self.parent))


@dataclass(frozen=True, eq=False)
class FunctionInvocationPrefix(AdmittedExecutionPrefixABC):
    """One original call followed by its distinct function-body activation.

    The call site and function declaration own the relation.  The parent cut
    ends at the invocation and the child begins at a fresh activation; source
    nesting or equal frame contents cannot manufacture this edge.
    """

    parent: AdmittedExecutionPrefixABC
    invocation: CompactFunctionCall
    declaration: CompactFunctionDeclaration
    child: AdmittedExecutionPrefixABC

    def __post_init__(self) -> None:
        parent = self.parent.endpoint
        child = self.child.intervals[0]
        if (
            not any(self.invocation is call for call in parent.context.flow.calls)
            or parent.position != self.invocation.position
        ):
            raise ValueError(
                "Function entry requires its original parent invocation cut"
            )
        if child.context.flow.owner is not self.declaration or child.after is not None:
            raise ValueError(
                "Function entry requires the callee declaration's fresh body interval"
            )

    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        pending.extend((self.child, self.parent))


class SourceDefinitionApplicationAuthorityABC(ABC):
    """One definition-owned decorator application and its exact completed cut."""

    @property
    @abstractmethod
    def application_prefix(self) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError

    @property
    @abstractmethod
    def definition_application(self) -> CompactMutation[CompactDefinitionTarget]:
        raise NotImplementedError

    @property
    @abstractmethod
    def decorator_use(self) -> CompactValueUse:
        raise NotImplementedError

    @abstractmethod
    def require_original_application(self) -> None:
        """Join source position and compiler operand topology for this application."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class DefinitionApplicationPrefix(AdmittedExecutionPrefixABC):
    """One original definition application followed by its callable activation."""

    application: SourceDefinitionApplicationAuthorityABC
    declaration: CompactFunctionDeclaration
    child: AdmittedExecutionPrefixABC

    def __post_init__(self) -> None:
        application = self.application
        application.require_original_application()
        parent = application.application_prefix.endpoint
        definition = application.definition_application
        child = self.child.intervals[0]
        if (
            not any(definition is event for event in parent.context.flow.mutations)
            or parent.position != definition.target.header_position
            or not any(
                application.decorator_use is use
                for use in definition.target.decorator_uses
            )
        ):
            raise ValueError(
                "Function entry requires its original definition application cut"
            )
        if child.context.flow.owner is not self.declaration or child.after is not None:
            raise ValueError(
                "Function entry requires the callee declaration's fresh body interval"
            )

    def _expand_intervals(
        self,
        pending: list[AdmittedExecutionPrefixABC],
        intervals: list[SingleFlowPrefix],
    ) -> None:
        pending.extend((self.child, self.application.application_prefix))


@dataclass(frozen=True)
class ContextualMutation:
    """An existing write observed in an admitted activation interval."""

    source: SingleFlowPrefix
    mutation: CompactMutation

    def same_installation(self, other: ContextualMutation) -> bool:
        return (
            self.source.context is other.source.context
            and self.source.frame is other.source.frame
            and self.mutation is other.mutation
        )


class CapturedReferenceEffectsABC(ABC):
    """Required complete contextual-prefix proof and actual activation relation.

    Admission supplies the actual locals/globals/captured-builtins relationship
    at this source/context/position. It closes every possibly preceding effect
    apart from direct namespace writes checked by the kernel: implicit operators,
    destruction, imports and hooks, star imports, class construction and calls.
    Missing compact records prove none of these. Initial exact-string-key
    namespace admission, retained object identities, captured sys.modules
    associations and native import behavior must remain valid at the query.
    Every interval belongs to one activation with fixed frame handles. The
    supplied prefix includes completed child executions and preserves their
    actual contexts; repeated activations must not share a canonical frame.
    Historical capture queries require their historical prefix, not the later
    observer's execution cut. Omitted direct writes are not closed by admission.

    Native name lookup and lexical write destinations must match the admitted
    frame. Deferred scopes, custom locals/closures and compiler-created frames
    require their own proof, never inference from source-parent containment.
    There is deliberately no permissive production implementation.
    """

    def entry_contents(
        self,
        kernel: CapturedReferenceKernel,
        namespace: NamespaceEvidenceABC,
        prefix: AdmittedExecutionPrefixABC,
    ) -> NamespaceContentsABC:
        """Storage before source operations, including admitted native entry work."""
        return CapturedEntryContents(kernel, namespace)

    def require_interval_effects(self, interval: SingleFlowPrefix) -> None:
        raise ValueError("Original interval effects remain unproved")

    def require_native_behavior(self, authority: SourceActivationAuthorityABC) -> None:
        """Require native effect behavior separately from mere successful completion."""
        raise ValueError("Native operation behavior needs an explicit entry condition")

    def require_operation_completion(
        self, authority: SourceActivationAuthorityABC
    ) -> None:
        """Require supplied completion at this cut, separately from binding/result."""
        raise ValueError(
            "Native operation completion needs an explicit entry condition"
        )

    def item_installation_value(
        self,
        occurrence: ContextualMutation,
    ) -> CapturedReferenceResolution:
        """Original storage execution is separate from ordinary prefix admission."""
        raise ValueError("Native item installation remains unproved")

    def call_result(
        self,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CapturedReferenceResolution:
        """Actual call execution and returned identity require their own admission."""
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)

    @abstractmethod
    def admit(
        self,
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> AdmittedExecutionPrefixABC | OpenCapturedReference:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class InitialNativeIsland:
    """Actual modules, import handles and shared initial dictionary admissions.

    Module dictionaries are derived from the admitted native modules. Additional
    actual frame storages are admitted at construction, never guessed at lookup.
    One namespace owner is captured per dictionary identity before any frame is
    assembled. Unregistered modules need not acquire import handles.
    """

    modules: tuple[ModuleType, ...]
    extra_storages: InitVar[tuple[dict[NativeScalar, object], ...]] = ()
    modules_by_name: Mapping[str, ModuleType] = field(init=False, repr=False)
    namespaces: tuple[NativeNamespace, ...] = field(init=False, repr=False)

    def require_frame(self, frame: InitialNativeFrame) -> None:
        """Validate exact evidence without demanding unused unresolved fallbacks."""
        for namespace in frame.namespace_values:
            if not isinstance(namespace, OpenCapturedReference):
                namespace.require_admitted(self)

    def __post_init__(
        self, extra_storages: tuple[dict[NativeScalar, object], ...]
    ) -> None:
        if any(type(module) is not ModuleType for module in self.modules):
            raise TypeError(
                "Initial native modules must have plain native module storage"
            )
        admitted_ids = {id(module) for module in self.modules}
        if len(admitted_ids) != len(self.modules):
            raise ValueError("Initial native module objects must be unique")
        storages = {
            id(storage): storage
            for storage in (*(vars(module) for module in self.modules), *extra_storages)
        }
        object.__setattr__(
            self,
            "namespaces",
            tuple(NativeNamespace(storage) for storage in storages.values()),
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

    def namespace_for_storage(
        self, storage: dict[NativeScalar, object]
    ) -> NativeNamespace:
        for namespace in self.namespaces:
            if namespace.storage is storage:
                return namespace
        raise ValueError("Namespace storage was not admitted by this initial island")

    def module(self, name: str | None) -> CapturedReferenceResolution:
        if name is None or name not in self.modules_by_name:
            return OpenCapturedReference(CapturedReferenceViolation.UNADMITTED_IMPORT)
        return CapturedNativeObject(self.modules_by_name[name])

    def require_registered_module_retention(
        self, name: str | None, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        """Use the prefix contract's preserved sys.modules association.

        This is a registered module reference, not generic module release safety
        or retention inferred from an analyzer-held object. Prefix admission
        independently requires the captured import associations to remain valid.
        """
        self.require_frame(prefix.endpoint.frame)
        self.module(name).require_closed()

    def imported_module(
        self, origin: ImportedNameOrigin
    ) -> CapturedReferenceResolution:
        bound_module = origin.qualified_name
        if (
            bound_module is None
            or origin.requested_module_name not in self.modules_by_name
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNADMITTED_IMPORT)
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

    def attribute_namespace(
        self,
        receiver: CapturedNativeObject,
        attribute: str,
    ) -> NativeNamespace | OpenCapturedReference:
        if type(attribute) is not str:
            raise TypeError("Native attribute access requires an exact string key")
        for module in self.modules:
            if receiver.value is module and not self._has_module_data_descriptor(
                attribute
            ):
                return self.namespace_for_storage(vars(module))
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        receiver: CapturedNativeObject,
        query: CapturedSlotQuery,
        occurrence: ContextualMutation,
        key: CapturedReferenceResolution,
    ) -> CapturedReferenceResolution | None:
        # Exact dict identity establishes distinct storage, not effect-free index
        # evaluation. Admission separately closes index/hash/destruction effects.
        if type(receiver.value) is not dict:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNKNOWN_RECEIVER, occurrence.mutation
            )
        if query.namespace.is_initial_storage(receiver.value):
            return query.item_key_write_effect(resolver, key, occurrence)
        return None


@dataclass(frozen=True)
class CapturedSlotQuery:
    """One namespace slot observed through a contextual admitted prefix."""

    namespace: NamespaceEvidenceABC
    key: NativeScalar
    prefix: AdmittedExecutionPrefixABC
    pending: frozenset[CompactBindingVisit[CompactFlowContext]]
    installed: ContextualMutation | None = None

    def matching_item_write(
        self,
        resolver: CapturedReferenceKernel,
        occurrence: ContextualMutation,
    ) -> CapturedReferenceResolution:
        """Effect-only queries retain their original conservative interference law."""
        return OpenCapturedReference(
            CapturedReferenceViolation.POSSIBLE_SLOT_WRITE, occurrence.mutation
        )

    def resolve(
        self, resolver: CapturedReferenceKernel
    ) -> CapturedReferenceResolution | None:
        """Select lexical installation and subsequent writes under this query's law."""
        self.namespace.require_key(self.key)
        for candidate in self.prefix.binding_sources(self.namespace, self.key):
            if isinstance(candidate, OpenCapturedReference):
                return candidate
            source, binding = candidate
            if binding.mutation is not None:
                query = replace(
                    self, installed=ContextualMutation(source, binding.mutation)
                )
                result = query.intervening_value(resolver)
                if result is not None:
                    return result
            return binding.resolve_binding(
                source.capture_kernel(resolver),
                source.context,
                LexicalValueReference(self.key),
                source.position,
                self.pending,
            )
        result = self.intervening_value(resolver)
        return (
            self.prefix.entry_contents(resolver, self.namespace).member(self.key)
            if result is None
            else result
        )

    def require_independent_reference(
        self,
        kernel: CapturedReferenceKernel,
        value: CapturedReferenceResolution,
    ) -> None:
        """Require another live slot retaining this object before this slot changes.

        The destination itself and duplicate frame handles cannot be witnesses.
        Namespace membership and values derive at this same admitted cut, never
        from historical alias spellings or the analyzer's retained references.
        """
        self.prefix.require_admitted(kernel.initial)
        value.require_closed()
        namespaces = {
            id(namespace): namespace
            for namespace in self.prefix.endpoint.frame.namespace_values
            if isinstance(namespace, NamespaceEvidenceABC)
        }
        for namespace in namespaces.values():
            for name in NamespaceMemberInventory(kernel, namespace, self.prefix).names:
                if namespace is self.namespace and name == self.key:
                    continue
                candidate = kernel._namespace_resolution(
                    namespace,
                    name,
                    self.prefix,
                    self.pending,
                )
                if candidate is not None and value.proves_same_object(candidate):
                    return
        raise ValueError("Object release has no proved independent retained reference")

    def mutations_after_installation(
        self,
    ) -> Iterator[ContextualMutation | OpenCapturedReference]:
        intervals = self.prefix.intervals
        anchor = -1
        if self.installed is not None:
            matches = tuple(
                index
                for index, interval in enumerate(intervals)
                if interval.context is self.installed.source.context
                and interval.frame is self.installed.source.frame
                and interval.contains(self.installed.mutation)
                and any(
                    self.installed.mutation is event
                    for event in interval.context.flow.mutations
                )
            )
            if len(matches) != 1:
                yield OpenCapturedReference(
                    CapturedReferenceViolation.UNPROVED_BINDING, self.installed.mutation
                )
                return
            anchor = matches[0]
        for index, interval in enumerate(intervals):
            if index < anchor:
                continue
            for occurrence in interval.mutation_occurrences():
                mutation = occurrence.mutation
                if self.installed is not None and index == anchor:
                    if occurrence.same_installation(self.installed):
                        continue
                    if not self.installed.mutation.position.may_precede(
                        mutation.position
                    ):
                        continue
                yield occurrence

    def intervening_value(
        self, resolver: CapturedReferenceKernel
    ) -> CapturedReferenceResolution | None:
        for occurrence in self.mutations_after_installation():
            if isinstance(occurrence, OpenCapturedReference):
                return occurrence
            mutation = occurrence.mutation
            visit = CompactBindingVisit(occurrence.source.context, mutation)
            if visit in self.pending and mutation.target.bound_name is None:
                return OpenCapturedReference(
                    CapturedReferenceViolation.CYCLIC_BINDING, mutation
                )
            query = replace(self, pending=self.pending | {visit})
            failure = mutation.resolve(resolver, (query, occurrence))
            if failure is not None:
                return failure
        return None

    def write_effect(
        self,
        namespace: NamespaceEvidenceABC,
        key: NativeScalar,
        mutation: CompactMutation,
    ) -> OpenCapturedReference | None:
        if namespace is self.namespace and key == self.key:
            return OpenCapturedReference(
                CapturedReferenceViolation.POSSIBLE_SLOT_WRITE, mutation
            )
        return None

    def item_key_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        key: CapturedReferenceResolution,
        occurrence: ContextualMutation,
    ) -> CapturedReferenceResolution | None:
        """Exact scalar contents select the query's law; unknown keys never select an installation."""
        try:
            name = key.require_native_scalar()
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.POSSIBLE_SLOT_WRITE,
                occurrence.mutation,
                error,
            )
        return (
            self.matching_item_write(resolver, occurrence) if name == self.key else None
        )


ImportQuery: TypeAlias = tuple[
    CompactFlowContext,
    CompactMutation,
    frozenset[CompactBindingVisit[CompactFlowContext]],
]


NamespaceMembershipContext: TypeAlias = tuple[ContextualMutation, set[NativeScalar]]


@dataclass(frozen=True)
class InstalledValueSlotQuery(CapturedSlotQuery):
    """Read one proved item installation without relaxing release admission."""

    def mutations_after_installation(
        self,
    ) -> Iterator[ContextualMutation | OpenCapturedReference]:
        """Prefer the latest original occurrence; later interference still wins.

        The shared prefix inventory supplies execution order and original cuts.
        An installed-value query need not recursively replay superseded stores.
        Effect-only queries retain their original conservative traversal.
        """
        return iter(reversed(tuple(super().mutations_after_installation())))

    def matching_item_write(
        self,
        resolver: CapturedReferenceKernel,
        occurrence: ContextualMutation,
    ) -> CapturedReferenceResolution:
        try:
            self.prefix.require_admitted(resolver.initial)
            actual = self.prefix.require_event(
                occurrence.source.context, occurrence.mutation, occurrence.source.frame
            )
            if actual is not occurrence.source:
                raise ValueError("Item installation has a foreign execution occurrence")
            suffix = replace(
                self,
                pending=self.pending
                - {CompactBindingVisit(occurrence.source.context, occurrence.mutation)},
                installed=occurrence,
            ).intervening_value(resolver)
            if suffix is not None:
                return suffix
            return resolver.effects.item_installation_value(occurrence)
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_BINDING,
                occurrence.mutation,
                error,
            )


class NamespaceContentsABC(ABC):
    """Complete contents of one admitted observation, not a new storage identity."""

    kernel: CapturedReferenceKernel

    @property
    @abstractmethod
    def names(self) -> frozenset[NativeScalar]:
        raise NotImplementedError

    @abstractmethod
    def member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        raise NotImplementedError

    @abstractmethod
    def require_closed(self) -> None:
        """Admit the content owner/cut, without asserting all member values closed."""
        raise NotImplementedError

    def require_member(self, key: NativeScalar) -> CapturedReferenceResolution:
        self.require_closed()
        value = self.member(key)
        if value is None:
            raise ValueError("Complete mapping member is absent")
        return value

    def require_same_mapping(self, other: NamespaceContentsABC) -> None:
        """Prove unordered contents, not dictionary identity or iteration order."""
        self.require_closed()
        other.require_closed()
        keys = self.names
        if keys != other.names:
            raise ValueError("Complete key membership changed")
        for key in keys:
            left, right = self.require_member(key), other.require_member(key)
            left.require_closed()
            right.require_closed()
            if not left.proves_same_object(right):
                raise ValueError("Mapping value identity remains unproved")


class InitialNamespaceContents(NamespaceContentsABC):
    """A view of supplied entry evidence, preserving its namespace identity."""

    kernel: CapturedReferenceKernel
    namespace: NamespaceEvidenceABC

    def require_closed(self) -> None:
        self.namespace.require_admitted(self.kernel.initial)

    @property
    def names(self) -> frozenset[NativeScalar]:
        self.require_closed()
        return self.namespace.initial_names

    def member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        self.require_closed()
        return self.namespace.member(key)


@dataclass(frozen=True)
class CapturedEntryContents(InitialNamespaceContents):
    kernel: CapturedReferenceKernel
    namespace: NamespaceEvidenceABC


@dataclass(frozen=True)
class NamespaceMemberInventory(
    CompactMutationResolverABC[NamespaceMembershipContext, None],
    NamespaceContentsABC,
):
    """Derived complete namespace contents at one admitted execution cut."""

    kernel: CapturedReferenceKernel
    namespace: NamespaceEvidenceABC
    prefix: AdmittedExecutionPrefixABC

    def member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        self.require_closed()
        return self.kernel._namespace_resolution(
            self.namespace, key, self.prefix, frozenset()
        )

    def require_closed(self) -> None:
        self.prefix.require_admitted(self.kernel.initial)
        self.namespace.require_available(self.kernel, self.prefix)

    @cached_property
    def names(self) -> frozenset[NativeScalar]:
        self.require_closed()
        names = set(self.prefix.entry_contents(self.kernel, self.namespace).names)
        for occurrence in self.prefix.mutation_occurrences():
            occurrence.mutation.resolve(self, (occurrence, names))
        return frozenset(names)

    def _write(
        self,
        context: NamespaceMembershipContext,
        namespace: NamespaceEvidenceABC | OpenCapturedReference,
        key: NativeScalar,
    ) -> None:
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
        if namespace is not self.namespace:
            return
        occurrence, names = context
        occurrence.source.require_definite(occurrence.mutation)
        occurrence.mutation.kind.binding_operation.update_namespace_members(names, key)

    def _binding_mutation_resolution(
        self,
        context: NamespaceMembershipContext,
        mutation: CompactMutation,
        name: str,
    ) -> None:
        occurrence, _ = context
        source = occurrence.source
        self._write(context, source.frame.binding_namespace(source.context, name), name)

    def _attribute_mutation_resolution(
        self,
        context: NamespaceMembershipContext,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> None:
        occurrence, _ = context
        kernel = occurrence.source.capture_kernel(self.kernel)
        receiver = kernel._read_use(
            mutation.target.receiver_use, occurrence.source.context, frozenset()
        )
        name = mutation.target.attribute_name
        self._write(context, receiver.attribute_namespace(kernel.initial, name), name)

    def _item_mutation_resolution(
        self,
        context: NamespaceMembershipContext,
        mutation: CompactMutation[CompactItemTarget],
    ) -> None:
        occurrence, _ = context
        kernel = occurrence.source.capture_kernel(self.kernel)
        receiver = kernel._read_use(
            mutation.target.receiver_use, occurrence.source.context, frozenset()
        )
        namespace = receiver.dictionary_namespace(kernel.initial)
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
        if namespace is not self.namespace:
            return
        key = kernel._read_use(
            mutation.target.index_use, occurrence.source.context, frozenset()
        ).require_native_scalar()
        self._write(context, namespace, key)

    def _receiver_mutation_resolution(
        self,
        context: NamespaceMembershipContext,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> None:
        raise ValueError("Complete receiver-write key membership remains unproved")


ValueQuery: TypeAlias = tuple[
    CompactFlowValue, frozenset[CompactBindingVisit[CompactFlowContext]]
]


@dataclass(frozen=True)
class CapturedReferenceKernel(
    CompactBindingResolverABC[CapturedReferenceResolution | None],
    CompactValueResolverABC[ValueQuery, CapturedReferenceResolution],
    ImportOriginResolverABC[ImportQuery, CapturedReferenceResolution],
    CompactMutationResolverABC[
        tuple[CapturedSlotQuery, ContextualMutation], CapturedReferenceResolution | None
    ],
):
    initial: InitialNativeIsland
    effects: CapturedReferenceEffectsABC

    def _compiler_stored_value_resolution(
        self, value: CompilerStoredValue, context: ValueQuery
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def _forwarded_result_value_resolution(
        self,
        value: ForwardedResultValue,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        """Forwarding requires the original read, result and storage execution proof."""
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def _deleted_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> None:
        """The selected admitted deletion proves absence in this exact namespace."""
        return None

    def _namespaces_resolution(
        self,
        prefix: AdmittedExecutionPrefixABC,
        namespaces: tuple[NamespaceEvidenceABC | OpenCapturedReference, ...],
        name: str,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        for namespace in namespaces:
            if isinstance(namespace, OpenCapturedReference):
                return namespace
            result = self._namespace_resolution(namespace, name, prefix, pending)
            if result is not None:
                return result
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def namespace_names(
        self,
        namespace: NamespaceEvidenceABC,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> frozenset[NativeScalar]:
        prefix = self._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        return NamespaceMemberInventory(self, namespace, prefix).names

    def _namespace_resolution(
        self,
        namespace: NamespaceEvidenceABC,
        key: NativeScalar,
        prefix: AdmittedExecutionPrefixABC,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution | None:
        """Resolve this exact storage slot through the ordinary value-query law."""
        return InstalledValueSlotQuery(namespace, key, prefix, pending).resolve(self)

    def read_source_value(
        self,
        source: SourceProductFlowProjection,
        node: ast.expr,
    ) -> CapturedReferenceResolution:
        """Request the actual evaluated value, never fall back to a lexical read."""
        capture = source.value_reads_by_node.get(node)
        if capture is None:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        return self.read(capture)

    def _call_result_value_resolution(
        self,
        value: CallResultValue,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, _ = context
        call = value.invocation
        if not any(
            call is candidate for candidate in read.context.flow.calls
        ) or not call.position.dominates(read.use.position):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        prefix = self._admitted_prefix(read.context, call.position)
        if isinstance(prefix, OpenCapturedReference):
            return prefix
        return self.effects.call_result(read.context, call)

    def _unproved_value_resolution(
        self, context: ValueQuery
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def _lexical_value_resolution(
        self,
        reference: LexicalValueReference,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, pending = context
        return self._read_reference(reference, read.context, read.use.position, pending)

    def read_source(
        self, source: SourceProductFlowProjection, node: ast.expr
    ) -> CapturedReferenceResolution:
        """Capture an original source operand through its canonical flow read.

        Equal coordinates, copied ASTs and ambiguous source-operation joins
        are insufficient. The effect authority still admits the actual context
        and historical cut through the ordinary read path.
        """
        read = source.reference_reads_by_node.get(node)
        if read is None:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        return self.read(read)

    def require_interval_frame(self, interval: SingleFlowPrefix) -> None:
        prefix = self._admitted_prefix(interval.context, interval.position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        if prefix.endpoint.frame is not interval.frame:
            raise ValueError("Interval capture requires its original activation frame")

    def _admitted_prefix(
        self,
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> AdmittedExecutionPrefixABC | OpenCapturedReference:
        prefix = self.effects.admit(context, position)
        if not isinstance(prefix, OpenCapturedReference):
            prefix.require_admitted(self.initial)
            if (
                prefix.endpoint.context is not context
                or prefix.endpoint.position != position
            ):
                raise ValueError(
                    "Admission must retain the actual requested context and cut"
                )
        return prefix

    def read(self, read: CompactFlowValue) -> CapturedReferenceResolution:
        return self._read_use(read.use, read.context, frozenset())

    def _read_use(
        self,
        use: CompactPositionedReference,
        context: CompactFlowContext,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return use.resolve_value(self, (CompactFlowValue(context, use), pending))

    def _read_reference(
        self,
        reference: LexicalValueReference,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        prefix = self._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            return prefix
        resolution = self._root_resolution(
            prefix, LexicalValueReference(reference.root_name), pending
        )
        for attribute in reference.attribute_path:
            resolution = resolution.access(self, attribute, context, position, pending)
        return resolution

    def _root_resolution(
        self,
        prefix: AdmittedExecutionPrefixABC,
        root: LexicalValueReference,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        endpoint = prefix.endpoint
        namespaces = endpoint.frame.initial_lookup_namespaces(
            endpoint.context, root.root_name
        )
        if isinstance(namespaces, OpenCapturedReference):
            return namespaces
        return self._namespaces_resolution(prefix, namespaces, root.root_name, pending)

    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.CYCLIC_BINDING)

    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNPROVED_BINDING, binding
        )

    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
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

    def _package_at_import(self, context: ImportQuery) -> str:
        flow_context, binding, pending = context
        prefix = self._admitted_prefix(flow_context, binding.position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        package = self._slot(
            prefix.endpoint.frame.globals,
            ImportFromModuleName.package_binding,
            flow_context,
            binding.position,
            pending,
        )
        if package is None:
            raise ValueError("Relative import package fallback remains unproved")
        return package.require_native_text()

    def member_import_module_name(
        self, origin: ImportedNameOrigin, context: ImportQuery
    ) -> str:
        (reference,) = origin.declaration.module_references
        return reference.resolve_from_package(partial(self._package_at_import, context))

    def _member_import_resolution(
        self,
        origin: ImportedNameOrigin,
        context: ImportQuery,
    ) -> CapturedReferenceResolution:
        flow_context, binding, pending = context
        return self.initial.module(
            self.member_import_module_name(origin, context)
        ).access(self, origin.alias.name, flow_context, binding.position, pending)

    def _slot(
        self,
        namespace: NamespaceEvidenceABC | OpenCapturedReference,
        key: NativeScalar,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution | None:
        if isinstance(namespace, OpenCapturedReference):
            return namespace
        prefix = self._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            return prefix
        return self._namespace_resolution(namespace, key, prefix, pending)

    def _binding_mutation_resolution(
        self,
        context: tuple[CapturedSlotQuery, ContextualMutation],
        mutation: CompactMutation,
        name: str,
    ) -> OpenCapturedReference | None:
        query, occurrence = context
        source = occurrence.source
        namespace = source.frame.binding_namespace(source.context, name)
        if isinstance(namespace, OpenCapturedReference):
            return namespace
        return query.write_effect(namespace, name, mutation)

    def _receiver_mutation_resolution(
        self,
        context: tuple[CapturedSlotQuery, ContextualMutation],
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> OpenCapturedReference:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
        )

    def _attribute_mutation_resolution(
        self,
        context: tuple[CapturedSlotQuery, ContextualMutation],
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        query, occurrence = context
        owner = occurrence.source.capture_kernel(self)
        receiver = owner._read_use(
            mutation.target.receiver_use, occurrence.source.context, query.pending
        )
        return receiver.write_effect(owner, query, mutation)

    def _item_mutation_resolution(
        self,
        context: tuple[CapturedSlotQuery, ContextualMutation],
        mutation: CompactMutation[CompactItemTarget],
    ) -> CapturedReferenceResolution | None:
        query, occurrence = context
        owner = occurrence.source.capture_kernel(self)
        receiver = owner._read_use(
            mutation.target.receiver_use, occurrence.source.context, query.pending
        )
        key = owner._read_use(
            mutation.target.index_use, occurrence.source.context, query.pending
        )
        return receiver.item_write_effect(owner, query, occurrence, key)
