"""Positioned object capture under an explicitly admitted native environment.

This kernel does not prove arbitrary Python execution effects. A mandatory
effect authority must close the actual source prefix before any capture is
accepted. Source-origin names alone never authenticate a runtime object.
"""

from __future__ import annotations

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
    CompactImportTarget,
    CompactItemTarget,
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
    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactItemTarget],
    ) -> OpenCapturedReference | None:
        raise NotImplementedError

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

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactItemTarget],
    ) -> OpenCapturedReference:
        return OpenCapturedReference(
            CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
        )

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

    def item_write_effect(
        self,
        resolver: CapturedReferenceKernel,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactItemTarget],
    ) -> OpenCapturedReference | None:
        return resolver.initial.item_write_effect(self, query, mutation)

    def access(
        self,
        resolver: CapturedReferenceKernel,
        attribute: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        namespace = resolver.initial.attribute_namespace(self, attribute)
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
        return resolver.initial.attribute_write_effect(self, query, mutation)

    def require_native_identity(
        self, declaration: NativeDeclaration
    ) -> NativeDeclaration:
        if self.value is not declaration.declaration:
            raise ValueError("Captured object is not the required native declaration")
        return declaration


@dataclass(frozen=True, eq=False)
class NativeNamespace:
    """One actual native dictionary and its admitted initial string-key state.

    Key admission precedes lookup/copy: even an exact dict can contain foreign
    keys whose equality runs code. The effect proof must preserve this invariant
    and account for later index, destruction and external mutation effects.
    """

    storage: dict[str, object]
    initial_entries: Mapping[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.storage) is not dict:
            raise TypeError("A native namespace requires exact dictionary storage")
        if any(type(key) is not str for key in self.storage):
            raise TypeError("A native namespace requires only exact string keys")
        object.__setattr__(
            self, "initial_entries", MappingProxyType(self.storage.copy())
        )

    def member(self, key: str) -> CapturedNativeObject | None:
        """None is proved initial absence, never an unproved lookup."""
        if type(key) is not str:
            raise TypeError("Native namespace lookup requires an exact string key")
        if key not in self.initial_entries:
            return None
        return CapturedNativeObject(self.initial_entries[key])


@dataclass(frozen=True, eq=False)
class InitialNativeFrame:
    """Actual admitted namespaces, not globals' mutable __builtins__ spelling.

    The source/position admission must prove these native lookup and write
    destinations apply to the current activation. Source parentage alone is
    insufficient, including compiler-created generic frames. Nonlocal binding
    requires a closure relation not represented by these namespace handles.
    """

    locals: NativeNamespace
    globals: NativeNamespace
    builtins: NativeNamespace

    def binding_namespace(
        self,
        context: CompactFlowContext,
        name: str,
    ) -> NativeNamespace | OpenCapturedReference:
        if name in context.flow.nonlocal_binding_names:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        if name in context.flow.global_binding_names:
            return self.globals
        return self.locals

    def initial_lookup_namespaces(
        self,
        context: CompactFlowContext,
        name: str,
    ) -> tuple[NativeNamespace, ...] | OpenCapturedReference:
        binding_namespace = self.binding_namespace(context, name)
        if isinstance(binding_namespace, OpenCapturedReference):
            return binding_namespace
        return tuple(dict.fromkeys((binding_namespace, self.globals, self.builtins)))


class CapturedReferenceEffectsABC(ABC):
    """Required complete source-prefix proof and actual activation frame.

    Admission supplies the actual locals/globals/captured-builtins relationship
    at this source/context/position. It closes every possibly preceding effect
    apart from direct namespace writes checked by the kernel: implicit operators,
    destruction, imports and hooks, star imports, class construction and calls.
    Missing compact records prove none of these. Initial exact-string-key
    namespace admission, retained object identities, captured sys.modules
    associations and native import behavior must remain valid at the query.
    The admitted current-flow prefix belongs to one activation with fixed frame
    namespace handles; repeated or contextual activations need an execution
    relation rather than flattening different frames into this prefix.

    Native name lookup and lexical write destinations must match the admitted
    frame. Deferred scopes, custom locals/closures and compiler-created frames
    require their own proof, never inference from source-parent containment.
    There is deliberately no permissive production implementation.
    """

    @abstractmethod
    def admit(
        self,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> InitialNativeFrame | OpenCapturedReference:
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
    extra_storages: InitVar[tuple[dict[str, object], ...]] = ()
    modules_by_name: Mapping[str, ModuleType] = field(init=False, repr=False)
    namespaces: tuple[NativeNamespace, ...] = field(init=False, repr=False)

    def require_frame(self, frame: InitialNativeFrame) -> None:
        """A frame reuses this admission's exact owners, not new snapshots."""
        for namespace in (frame.locals, frame.globals, frame.builtins):
            if not any(namespace is admitted for admitted in self.namespaces):
                raise ValueError(
                    "Frame namespace owner belongs to a different admission"
                )

    def __post_init__(self, extra_storages: tuple[dict[str, object], ...]) -> None:
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

    def namespace_for_storage(self, storage: dict[str, object]) -> NativeNamespace:
        for namespace in self.namespaces:
            if namespace.storage is storage:
                return namespace
        raise ValueError("Namespace storage was not admitted by this initial island")

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
        for module in self.modules:
            if receiver.value is module and not self._has_module_data_descriptor(
                attribute
            ):
                return self.namespace_for_storage(vars(module))
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_ACCESS)

    def attribute_write_effect(
        self,
        receiver: CapturedNativeObject,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> OpenCapturedReference | None:
        namespace = self.attribute_namespace(receiver, mutation.target.attribute_name)
        if isinstance(namespace, OpenCapturedReference):
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, mutation
            )
        return query.write_effect(namespace, mutation.target.attribute_name, mutation)

    def item_write_effect(
        self,
        receiver: CapturedNativeObject,
        query: CapturedSlotQuery,
        mutation: CompactMutation[CompactItemTarget],
    ) -> OpenCapturedReference | None:
        # Exact dict identity establishes distinct storage, not effect-free index
        # evaluation. Admission separately closes index/hash/destruction effects.
        if type(receiver.value) is not dict:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNKNOWN_RECEIVER, mutation
            )
        if receiver.value is query.namespace.storage:
            # Compact item indices do not yet retain a proved exact string key.
            return OpenCapturedReference(
                CapturedReferenceViolation.POSSIBLE_SLOT_WRITE, mutation
            )
        return None


@dataclass(frozen=True)
class CapturedSlotQuery:
    """An actual dictionary-slot obligation and its positioned source context."""

    namespace: NativeNamespace
    key: str
    context: CompactFlowContext
    frame: InitialNativeFrame
    pending: frozenset[CompactBindingVisit]

    installed: CompactMutation | None = None

    def failure_before(
        self,
        resolver: CapturedReferenceKernel,
        position: CompactFlowPosition,
    ) -> OpenCapturedReference | None:
        for mutation in self.mutations_before(position):
            visit = (self.context.owner_symbol, mutation)
            if visit in self.pending and mutation.target.bound_name is None:
                return OpenCapturedReference(
                    CapturedReferenceViolation.CYCLIC_BINDING, mutation
                )
            query = replace(self, pending=self.pending | {visit})
            failure = mutation.resolve(resolver, query)
            if failure is not None:
                return failure
        return None

    def mutations_before(
        self,
        position: CompactFlowPosition,
    ) -> Iterator[CompactMutation]:
        """Retain any write possibly between installation and this actual use.

        The selected installation is not an intervening write. Earlier writes are
        excluded only by the shared positioned proof, never source line sorting;
        loop and unordered-header possibilities consequently remain obligations.
        """
        for mutation in self.context.flow.mutations:
            if mutation is self.installed or not mutation.position.may_precede(
                position
            ):
                continue
            if self.installed is not None and not self.installed.position.may_precede(
                mutation.position
            ):
                continue
            yield mutation

    def write_effect(
        self,
        namespace: NativeNamespace,
        key: str,
        mutation: CompactMutation,
    ) -> OpenCapturedReference | None:
        if namespace.storage is self.namespace.storage and key == self.key:
            return OpenCapturedReference(
                CapturedReferenceViolation.POSSIBLE_SLOT_WRITE, mutation
            )
        return None


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

    def _selected_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution:
        if use_position is None:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_BINDING, binding
            )
        frame = self._admitted_frame(context, use_position)
        if isinstance(frame, OpenCapturedReference):
            return frame
        namespace = frame.binding_namespace(context, reference.root_name)
        if isinstance(namespace, OpenCapturedReference):
            return namespace
        failure = CapturedSlotQuery(
            namespace, reference.root_name, context, frame, pending_bindings, binding
        ).failure_before(self, use_position)
        if failure is not None:
            return failure
        return super()._selected_binding_resolution(
            context, reference, binding, use_position, pending_bindings
        )

    def _admitted_frame(
        self,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> InitialNativeFrame | OpenCapturedReference:
        frame = self.effects.admit(context, position)
        if not isinstance(frame, OpenCapturedReference):
            self.initial.require_frame(frame)
        return frame

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
        frame = self._admitted_frame(context, position)
        if isinstance(frame, OpenCapturedReference):
            return frame
        root = LexicalValueReference(reference.root_name)
        binding = context.flow.binding_resolution_for(root.root_name, position)
        if binding is None:
            namespaces = frame.initial_lookup_namespaces(context, root.root_name)
            if isinstance(namespaces, OpenCapturedReference):
                return namespaces
            resolution: CapturedReferenceResolution = OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_BINDING
            )
            for namespace in namespaces:
                result = self._slot(
                    namespace, root.root_name, context, position, pending
                )
                if result is not None:
                    resolution = result
                    break
        else:
            resolution = binding.resolve_binding(self, context, root, position, pending)
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
        namespace: NativeNamespace,
        key: str,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        pending: frozenset[CompactBindingVisit],
    ) -> CapturedReferenceResolution | None:
        frame = self._admitted_frame(context, position)
        if isinstance(frame, OpenCapturedReference):
            return frame
        failure = CapturedSlotQuery(
            namespace, key, context, frame, pending
        ).failure_before(self, position)
        return namespace.member(key) if failure is None else failure

    def _binding_mutation_resolution(
        self,
        context: CapturedSlotQuery,
        mutation: CompactMutation,
        name: str,
    ) -> OpenCapturedReference | None:
        # The admitted prefix shares one actual activation's fixed frame handles.
        namespace = context.frame.binding_namespace(context.context, name)
        if isinstance(namespace, OpenCapturedReference):
            return namespace
        return context.write_effect(namespace, name, mutation)

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
    ) -> OpenCapturedReference | None:
        receiver = self._read_use(
            mutation.target.receiver_use, context.context, context.pending
        )
        return receiver.item_write_effect(self, context, mutation)
