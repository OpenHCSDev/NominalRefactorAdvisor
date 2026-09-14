"""Explicit analyzed-program entry premises for source namespace evidence.

These are supplied execution premises, not observations of a target process.
Source bindings are interpreted from their actual ordered flow, never copied
into initial namespace facts. A prefix effect proof is still required for any
native-reference admission after entry.
"""

from __future__ import annotations

import builtins
import dataclasses
import typing

from abc import abstractmethod
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import InitVar, dataclass, field
from functools import cached_property, cache
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from types import MappingProxyType

from .captured_reference import (
    AdmittedExecutionPrefixABC,
    CapturedNativeObject,
    CapturedReferenceResolution,
    CreatedNamespaceDictionary,
    InitialNativeFrame,
    InitialNativeIsland,
    NamespaceEvidenceABC,
    NativeNamespace,
    NativeTypePremise,
    OpenCapturedReference,
    RecordedNamespace,
    CapturedFlowPrefix,
    CapturedReferenceKernel,
    SourceActivationAuthorityABC,
)
from .collection_algebra import UniqueIdentityIndexAuthority
from .descriptor_algebra import AliasProperty
from .native_declarations import (
    NativeDeclarationFamily,
    NativeLanguageFeature,
    NativeScalar,
)
from .product_flow import (
    CompactFlowContext,
    CompactFlowPosition,
    SourceFlowOperation,
    SourceProductFlowProjection,
)


@cache
def _native_exec_bootstrap_binding() -> str:
    """Derive the interpreter's empty-exec insertion, never run target source."""
    namespace: dict[str, object] = {}
    exec(compile("", "<native-source-entry-premise>", "exec"), namespace)
    if len(namespace) != 1:
        raise ValueError("Native source entry has an unproved bootstrap convention")
    return next(iter(namespace))


@dataclass(frozen=True)
class DeclaredOperationCompletion:
    """Supplied successful completion, without a restriction on callback effects."""

    operation: SourceFlowOperation

    def require_native_behavior(self, protocol: type[NativeDeclarationFamily]) -> None:
        raise ValueError("Completion alone does not establish native effect behavior")


@dataclass(frozen=True)
class DeclaredNativeOperationBehavior(DeclaredOperationCompletion):
    """Explicit completion with only the selected native protocol's declared effects.

    This supported-execution premise covers the actual implementation and its
    dependency callbacks at the original invocation. They do not mutate admitted
    source namespaces, retained inputs or native associations outside the effects
    declared by the selected protocol. It is not an observed execution, argument
    binding proof, returned-object identity or a requested output mapping.
    """

    protocol: type[NativeDeclarationFamily]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, type) or not issubclass(
            self.protocol, NativeDeclarationFamily
        ):
            raise TypeError("Native behavior requires its nominal protocol owner")

    def require_native_behavior(self, protocol: type[NativeDeclarationFamily]) -> None:
        if protocol is not self.protocol:
            raise ValueError("Native behavior condition belongs to another protocol")


class SourceExecutionEntryABC(NamespaceEvidenceABC):
    """Original entry namespace and frame of one source activation.

    Entry storage is not necessarily a native dictionary. Each activation kind
    supplies its original prefix and conditions; a frame is not an effect proof.
    """

    source: SourceProductFlowProjection
    initial: InitialNativeIsland

    bootstrap_binding_name = staticmethod(_native_exec_bootstrap_binding)

    @property
    @abstractmethod
    def context(self) -> CompactFlowContext:
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> InitialNativeFrame:
        raise NotImplementedError

    def require_context(self, context: CompactFlowContext) -> None:
        if context is not self.context:
            raise ValueError(
                "Source activation remains unproved: entry requires its actual flow context"
            )

    @abstractmethod
    def prefix(
        self, position: CompactFlowPosition | None, kernel: CapturedReferenceKernel
    ) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError

    @abstractmethod
    def require_external_noninterference(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def require_native_behavior(self, authority: SourceActivationAuthorityABC) -> None:
        raise NotImplementedError

    @abstractmethod
    def require_operation_completion(
        self, authority: SourceActivationAuthorityABC
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class SourceModuleEntryPremise(
    CreatedNamespaceDictionary, RecordedNamespace, SourceExecutionEntryABC
):
    """One declared fresh module activation with exact native dictionary storage.

    Exact dictionary storage is part of this supplied execution premise,
    independently of the analyzer's initial-binding fact table.
    The caller supplies the target's entry convention and complete binding
    evidence: an absent key asserts initial absence; an existing unknown value
    must be an explicit OpenCapturedReference. The dictionary is a fact table,
    not a fabricated runtime globals dictionary. Captured builtins are supplied
    independently of any initial global spelling of ``__builtins__``.

    The namespace is declared fresh after the native island was captured. Each
    premise owns that activation's canonical namespace and frame; structurally
    equal source projections or separately constructed premises do not identify
    the same activation. This premise alone admits no preceding source effects.

    Optional declared operation conditions distinguish successful completion
    from native behavior with only its declared effects. They belong to original
    invocation cuts and never bypass independent binding/prefix checks or prove
    returned identity. The immutable condition index is normalized once from
    explicit inputs; default loaders never infer or transport these premises.
    """

    source: SourceProductFlowProjection
    native_island: InitialNativeIsland
    bindings: InitVar[dict[NativeScalar, CapturedReferenceResolution]]
    builtins: NamespaceEvidenceABC | OpenCapturedReference
    initial_entries: Mapping[NativeScalar, CapturedReferenceResolution] = field(
        init=False, repr=False
    )

    initial = AliasProperty[InitialNativeIsland]("native_island")

    context = AliasProperty[CompactFlowContext]("source.module_context")

    declared_operation_conditions: InitVar[Iterable[DeclaredOperationCompletion]] = (
        field(default=(), kw_only=True)
    )
    operation_conditions: Mapping[SourceFlowOperation, DeclaredOperationCompletion] = (
        field(init=False, repr=False)
    )

    def require_external_noninterference(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        """Source effect admission alone does not constrain external instrumentation."""
        raise ValueError("External source interference remains unproved")

    def require_native_behavior(self, authority: SourceActivationAuthorityABC) -> None:
        """Consume the actual invocation's explicitly selected native effect contract."""
        self._operation_condition(authority).require_native_behavior(type(authority))

    def require_operation_completion(
        self, authority: SourceActivationAuthorityABC
    ) -> None:
        """Require successful completion without upgrading it to an effect contract."""
        self._operation_condition(authority)

    def _operation_condition(
        self, authority: SourceActivationAuthorityABC
    ) -> DeclaredOperationCompletion:
        """Resolve one condition after validating its canonical source activation cut."""
        authority.require_condition_activation(self)
        try:
            return self.operation_conditions[authority.operation]
        except KeyError as error:
            raise ValueError(
                "Native operation needs an explicit entry condition"
            ) from error

    def __post_init__(
        self,
        bindings: dict[NativeScalar, CapturedReferenceResolution],
        declared_operation_conditions: Iterable[DeclaredOperationCompletion],
    ) -> None:
        conditions = tuple(declared_operation_conditions)
        for condition in conditions:
            if not isinstance(condition, DeclaredOperationCompletion):
                raise TypeError(
                    "Operation conditions require nominal premise declarations"
                )
            operation = condition.operation
            if self.source.event_operation(operation.event) is not operation:
                raise ValueError(
                    "Operation conditions require original source operations"
                )
        index = UniqueIdentityIndexAuthority.declarations_by_handle(
            conditions, lambda condition: condition.operation
        )
        object.__setattr__(self, "operation_conditions", MappingProxyType(index))
        entries = self.capture_initial_entries(bindings)
        if any(
            not isinstance(value, CapturedReferenceResolution)
            for value in entries.values()
        ):
            raise TypeError("Source initial bindings require explicit value evidence")
        object.__setattr__(self, "initial_entries", entries)
        _ = self.context  # Validate and retain the unique source module context.
        self.initial.require_frame(self.frame)

    @cached_property
    def frame(self) -> InitialNativeFrame:
        return InitialNativeFrame(self, self, self.builtins)

    def prefix(
        self, position: CompactFlowPosition | None, kernel: CapturedReferenceKernel
    ) -> AdmittedExecutionPrefixABC:
        return CapturedFlowPrefix(self.context, self.frame, position, kernel=kernel)

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if initial is not self.initial:
            raise ValueError("Source namespace belongs to a different entry premise")

    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        return self.initial_entries.get(key)


class NoninterferingSourceModuleEntryPremise(SourceModuleEntryPremise):
    """Explicitly supplied exclusion of external interference during activation.

    External tracing, debugging, other threads and asynchronous instrumentation
    do not change source storage or retained objects during this execution.
    Invoked source/native callbacks still need their ordinary effect proofs.
    Default loaders do not infer this premise or transfer it to another entry.
    """

    def require_external_noninterference(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        prefix.require_admitted(self.initial)
        for interval in prefix.intervals:
            if interval.frame.globals is not self:
                raise ValueError(
                    "External scope belongs to a different source activation"
                )
            if (
                self.source.context_for_owner(interval.context.flow.owner)
                is not interval.context
            ):
                raise ValueError("External scope requires its original source contexts")


class RegisteredSourceModuleEntryPremise(SourceModuleEntryPremise):
    """A fresh entry registered under its convention's actual runtime name.

    Registration is distinct from the source index's nominal module identity.
    Its collision obligation applies only after source/frame evidence validates.
    """

    @property
    @abstractmethod
    def registration_name(self) -> str:
        raise NotImplementedError

    def __post_init__(
        self,
        bindings: dict[NativeScalar, CapturedReferenceResolution],
        declared_operation_conditions: Iterable[DeclaredOperationCompletion],
    ) -> None:
        super().__post_init__(bindings, declared_operation_conditions)
        if self.registration_name in self.initial.modules_by_name:
            raise ValueError(
                "Fresh source entry would replace an admitted native module association"
            )


class ImportedSourceModuleEntryPremise(RegisteredSourceModuleEntryPremise):
    """Standard source import registers the source module's qualified name."""

    registration_name = AliasProperty[str]("source.module.module_name")

    @classmethod
    def from_source(
        cls, source: SourceProductFlowProjection
    ) -> ImportedSourceModuleEntryPremise:
        """Declare cached standard-library imports, not permission to execute them.

        Native calls still require their own behavior and argument evidence.
        Custom loaders and fresh imports use an explicitly supplied entry premise.
        """
        initial = InitialNativeIsland(
            (builtins, typing, dataclasses, NativeLanguageFeature.module)
        )
        return cls.from_standard_source_loader(
            source,
            initial,
            initial.namespace_for_storage(vars(builtins)),
        )

    @classmethod
    def from_standard_source_loader(
        cls,
        source: SourceProductFlowProjection,
        initial: InitialNativeIsland,
        builtins: NativeNamespace,
    ) -> ImportedSourceModuleEntryPremise:
        """Declare a fresh standard source-import activation, not observe one.

        Native loader metadata determines initial keys and exact value types.
        Its sample objects are not target identities; other protocols stay open.
        The caller supplies the admitted builtin dictionary association. Custom
        loaders, reloads and prepopulated exec namespaces need another premise.
        """
        module = source.module
        builtins.require_admitted(initial)
        loader = SourceFileLoader(module.module_name, module.file_path)
        spec = spec_from_file_location(
            module.module_name,
            module.file_path,
            loader=loader,
            submodule_search_locations=[] if module.is_package_init else None,
        )
        if spec is None:
            raise ValueError("Standard source loader did not establish module entry")
        template = module_from_spec(spec)
        bindings: dict[NativeScalar, CapturedReferenceResolution] = {
            name: NativeTypePremise(type(value))
            for name, value in vars(template).items()
        }
        bindings[cls.bootstrap_binding_name()] = CapturedNativeObject(builtins.storage)
        return cls(source, initial, bindings, builtins)


class DirectScriptEntryPremise(RegisteredSourceModuleEntryPremise):
    """Explicit direct-file execution registered as the interpreter's main module.

    The caller supplies complete initial binding facts for this interpreter and
    entry convention, plus its independent builtin association. No runpy or
    source-loader metadata substitutes for direct-file startup. The source's
    nominal module name does not replace that module's admitted import handle.
    Cached module associations restrict supported entries; they do not prove
    arbitrary cold imports, startup hooks or later callable behavior.
    """

    registration_name = "__main__"
