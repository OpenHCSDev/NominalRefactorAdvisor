"""Source-prefix native admission under an explicit fresh-source entry premise."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import dataclass, field
from functools import cached_property
from types import (
    CellType,
    FunctionType,
    MappingProxyType,
)
from typing import ClassVar, Generic, TypeVar, cast

from .ast_projection import AstClassProjection

from .ast_tools import ParsedModule, module_syntax_index
from .call_binding import (
    CompactCallArgument,
    CompactCallBinding,
    CompactFunctionSignature,
)
from .captured_reference import (
    AdmittedExecutionPrefixABC,
    CapturedFlowPrefix,
    CapturedEntryContents,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceResolution,
    CapturedReferenceViolation,
    CapturedSlotQuery,
    ChildExecutionPrefix,
    ContextualMutation,
    DefinitionApplicationPrefix,
    EmptyDictionaryCreation,
    FunctionInvocationPrefix,
    InitialNamespaceContents,
    InitialNativeFrame,
    InitialNativeIsland,
    InstanceProtocolEvidenceABC,
    NamespaceContentsABC,
    NamespaceCreationEvidenceABC,
    NamespaceEvidenceABC,
    NamespaceMemberInventory,
    CapturedNativeObject,
    NativeTypeCapture,
    NativeTypePremise,
    OpaqueCapturedObjectOperations,
    OpenCapturedReference,
    RecordedNamespace,
    SequentialExecutionPrefix,
    SingleFlowPrefix,
    SourceActivationAuthorityABC,
    SourceDefinitionApplicationAuthorityABC,
    ValueQuery,
)
from .class_namespace import SourceExecutionEffectEvidence
from .class_mro import DeclarationMroType
from .descriptor_algebra import AliasProperty
from .lexical_bindings import (
    FunctionParameterSource,
    ImportedNameOrigin,
    ImportOriginResolverABC,
)
from .native_class_mro import NativeClassMroDeclaration
from .native_compilation import (
    CPythonClassConstructionField,
    ExactNativeClassCapture,
    ExactNativeClassPrologue,
    ExactNativeFunctionExecution,
    ModuleNativeFrameOrigin,
    NativeAttributeValue,
    NativeAnnotationNamespaceValue,
    NativeBindingTransfer,
    NativeBindingTransferResolverABC,
    NativeCallValue,
    NativeCaptureSite,
    NativeClassClosureValue,
    NativeLocalValue,
    NativeClassCaptureResolverABC,
    NativeClassPrologueResolverABC,
    NativeConstantStore,
    NativeDefinitionApplication,
    NativeCompletionResolverABC,
    NativeCreationBackend,
    NativeDiscardValue,
    NativeImportValue,
    NativeImportMemberValue,
    NativeItemStoreValue,
    NativeListValue,
    NativeEmptyDictionaryValue,
    NativeFrameOriginResolverABC,
    NativeFunctionExecution,
    NativeFunctionValue,
    NativeNameValue,
    NativeProducedValue,
    NativeReadValue,
    NativeReturn,
    NativeStackEffectABC,
    NativeSubscriptionValue,
    NativeTupleValue,
    NativeTypedValue,
    NativeValueResolverABC,
    NativeValueStore,
    GeneratedClassNativeFrameOrigin,
    OpenNativeClassCapture,
    OpenNativeClassPrologue,
    OpenNativeFrameOrigin,
    SourceNativeFrameOrigin,
)
from .native_declarations import (
    NativeDeclaration,
    NativeDeclarationFamily,
    NativeScalar,
    QualifiedDeclaration,
)
from .native_reference import NativeReferenceEnvironment
from .native_call import (
    CallAuthority,
    DefaultObjectConstruction,
    NativeDefinitionApplicationAuthorityABC,
    NativeDescriptorArgumentABC,
    NativeDescriptorResult,
    SignatureCallAuthorityABC,
)
from .product_flow import (
    CompactAttributeTarget,
    CompactBranchPredicateResolverABC,
    CompactBindingValueResolverABC,
    CompactBindingVisit,
    CompactCallableReferenceUse,
    CompactClassDeclaration,
    CompactCompilerOperandUse,
    CompactDefinitionResolverABC,
    CompactDefinitionSource,
    CompactDefinitionTarget,
    CompactEvaluatedAssignment,
    CompactEvaluatedResult,
    CompactFlowContext,
    CompactFlowPosition,
    CompactFlowValue,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactFunctionTargetResolutionViolation,
    CompactImportTarget,
    CompactItemTarget,
    CompactLexicalBindingTargetABC,
    CompactMutation,
    CompactMutationResolverABC,
    CompactResultCompletionResolverABC,
    CompactNativeCapture,
    CompactPositionedReference,
    CompactSubscription,
    CompactTupleValue,
    CompactValueUse,
    CompactValueDestinationKind,
    CompilerOperandValue,
    CompilerStoredValue,
    FlowFrameResolverABC,
    ForwardedResultValue,
    InitialCompactParameterBinding,
    SourceFlowEvaluation,
    SourceFlowEvent,
    SourceFlowOperation,
    SourceProductFlowProjection,
    SubscriptionResultValue,
    source_product_flow_projection,
)
from .registry_identity import AutoRegisterClassAuthority
from .source_entry import (
    ImportedSourceModuleEntryPremise,
    SourceExecutionEntryABC,
    SourceModuleEntryPremise,
)
from .source_geometry import SourceByteSpan
from .value_expression import (
    EmptyDictionaryExpression,
    FunctionExpression,
    LexicalValueReference,
    LiteralExpressionEffects,
    OpaqueValueExpression,
)

SourceDefinitionNodeT = TypeVar(
    "SourceDefinitionNodeT", bound=ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
)


class SourceNativeFrameResolver(NativeFrameOriginResolverABC[CompactFlowContext]):
    """Match native origin to the actual admitted activation at a source cut."""

    execution: SourceExecutionABC

    def require_event_available_in(
        self, prefix: AdmittedExecutionPrefixABC, event: SourceFlowEvent
    ) -> None:
        """Require an original event in this frame at its canonical completed cut."""
        endpoint = prefix.endpoint
        observer = endpoint.capture_kernel(self.execution.kernel)
        if observer._admitted_prefix(endpoint.context, endpoint.position) is not prefix:
            raise ValueError("Source event requires the canonical closed source cut")
        prefix.require_event(
            self.native_frame_context, event, self.native_frame_prefix.endpoint.frame
        )

    def require_complete_source_cut(self, prefix: AdmittedExecutionPrefixABC) -> None:
        if prefix is not self.execution.required_prefix(
            self.native_frame_context, None
        ):
            raise ValueError(
                "Native continuation requires the complete original source cut"
            )

    @property
    @abstractmethod
    def native_frame_context(self) -> CompactFlowContext:
        raise NotImplementedError

    @property
    @abstractmethod
    def native_frame_prefix(self) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError

    def _module_frame_origin_resolution(
        self,
        origin: ModuleNativeFrameOrigin,
    ) -> CompactFlowContext:
        context = self.native_frame_context
        if (
            context is not self.execution.source.module_context
            or origin.compilation
            is not self.execution.module.native_compilation.identity
        ):
            raise ValueError("Native origin belongs to a different module activation")
        return context

    def _source_frame_origin_resolution(
        self,
        origin: SourceNativeFrameOrigin,
    ) -> CompactFlowContext:
        endpoint = self.native_frame_prefix.endpoint
        namespace = endpoint.frame.locals
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
        namespace.require_native_creator(origin.execution)
        return endpoint.context

    def _generated_class_frame_origin_resolution(
        self, origin: GeneratedClassNativeFrameOrigin
    ) -> CompactFlowContext:
        raise ValueError("Generated class frame has no matching source entry")

    def _open_frame_origin_resolution(
        self,
        origin: OpenNativeFrameOrigin,
    ) -> CompactFlowContext:
        raise ValueError("Native origin has no admitted executing frame")


class SourceCompletedReturnABC(SourceNativeFrameResolver):
    """An original completed source operation with a native return continuation."""

    @property
    @abstractmethod
    def operation(self) -> SourceFlowOperation:
        raise NotImplementedError

    @abstractmethod
    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        raise NotImplementedError

    @abstractmethod
    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        raise NotImplementedError

    @property
    def completion_evaluation(self) -> SourceFlowEvaluation:
        operation = self.operation
        source = self.execution.source
        if source.event_operation(operation.event) is not operation:
            raise ValueError("Completion evaluation requires its original operation")
        candidates = tuple(
            evaluation
            for evaluation in source.evaluation_bounds_by_node.get(operation.node, ())
            if evaluation.owner is operation.owner
        )
        if len(candidates) != 1:
            raise ValueError("Completion has no unique original evaluation")
        return candidates[0]


class SourceInstalledReturnABC(SourceCompletedReturnABC):
    """Source completions whose native boundary installs an original binding."""

    @abstractmethod
    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        raise NotImplementedError

    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        return self.require_native_installation(prefix).instruction_offset


class SourceNativeProductionABC(ABC):
    """Source execution owning an authenticated native production inventory."""

    execution: SourceExecutionABC

    @abstractmethod
    def _require_native_production(self, value: NativeProducedValue) -> None:
        raise NotImplementedError


class SourceNativeOperandInventoryABC(SourceNativeProductionABC):
    """An operand inventory with its source completion cut, not an expression."""

    @property
    @abstractmethod
    def source_completion_prefix(self) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class SourceDefinitionEntry(
    SourceActivationAuthorityABC,
    SourceNativeFrameResolver,
    Generic[SourceDefinitionNodeT],
):
    """Original source definition and its enclosing activation obligation.

    Source ancestry is not native activation. Module and class-body creators
    require their actual admitted frame; function activations and generated
    creators retain their separate open obligations.
    """

    execution: SourceExecutionABC
    node: SourceDefinitionNodeT

    native_frame_context = AliasProperty[CompactFlowContext]("parent_context")

    parent_prefix = AliasProperty[AdmittedExecutionPrefixABC]("native_frame_prefix")

    environment = AliasProperty[NativeReferenceEnvironment]("execution")

    activation_context = AliasProperty[CompactFlowContext]("parent_context")

    activation_position = AliasProperty[CompactFlowPosition](
        "definition.target.header_position"
    )

    activation_prefix = AliasProperty[AdmittedExecutionPrefixABC]("parent_prefix")

    @property
    def native_result_store(self) -> NativeValueStore:
        """Select the original final binding without granting result identity."""
        decorators = self.decorator_nodes
        production = decorators[0] if decorators else self.node
        store = self.execution.module.native_compilation.value_store_for(
            SourceByteSpan.require_node(production),
            SourceByteSpan.require_node(self.node),
            self.definition.target.bound_name,
        )
        store.frame.resolve(self)
        return store

    @property
    def decorator_nodes(self) -> tuple[ast.expr, ...]:
        """Rejoin the original decorator operands and their current source nodes."""
        self.require_original_operation()
        nodes = self.node.decorator_list
        uses = self.definition.target.decorator_uses
        if len(nodes) != len(uses) or any(
            self.execution.source_operation(self.parent_context, use).node is not node
            for node, use in zip(nodes, uses)
        ):
            raise ValueError(
                "Definition decorators differ from their original source chain"
            )
        return tuple(nodes)

    def require_operation_kind(self) -> None:
        """Rejoin the actual definition node even after its operation was cached."""
        if (
            self.execution.definition_operation(self.node) is not self.operation
            or self.definition is not self.operation.event
        ):
            raise ValueError(
                "Definition requires its original canonical source operation"
            )

    @cached_property
    def native_frame_prefix(self) -> AdmittedExecutionPrefixABC:
        return self.execution.required_prefix(
            self.parent_context,
            self.definition.target.header_position,
        )

    @cached_property
    def operation(self) -> SourceFlowOperation:
        return self.execution.definition_operation(self.node)

    @cached_property
    def definition(self) -> CompactMutation[CompactDefinitionTarget]:
        return cast(CompactMutation[CompactDefinitionTarget], self.operation.event)

    @cached_property
    def context(self) -> CompactFlowContext:
        return self.execution.context_for_owner(self.definition.target.owner)

    @cached_property
    def parent_context(self) -> CompactFlowContext:
        return self.execution.context_for_owner(self.operation.owner)

    @property
    def creation_results(self) -> tuple[SourceDefinitionResultABC, ...]:
        """Definition declarations own their canonical result application chain."""
        raise ValueError("Definition result application chain remains unproved")


class SourceDefinitionResultABC(CapturedReferenceResolution, SourceInstalledReturnABC):
    """Availability of a definition result, independent of raw object identity."""

    creation: SourceDefinitionEntry

    def require_available_in(self, prefix: AdmittedExecutionPrefixABC) -> None:
        """Authenticate a supplied canonical cut containing this original definition."""
        creation = self.creation
        creation.require_original_operation()
        self.require_closed()
        creation.require_event_available_in(prefix, creation.definition)

    def require_available_at(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition | None,
    ) -> AdmittedExecutionPrefixABC:
        """Require the installed original definition before a canonical source cut."""
        prefix = resolver._admitted_prefix(context, position)
        if isinstance(prefix, OpenCapturedReference):
            prefix.require_closed()
        self.require_available_in(prefix)
        return prefix

    def require_installation_name(
        self, installation: NativeBindingTransfer
    ) -> NativeBindingTransfer:
        if installation.name != self.creation.definition.target.bound_name:
            raise ValueError(
                "Native installation differs from its original source binding"
            )
        return installation


class SourceDefinitionCapture(SourceDefinitionResultABC):
    """Identity of one closed, non-repeated source definition creation.

    The creation owner includes the supplied activation, not just its AST.
    Future repeated activations need per-creation evidence before admission.
    """

    def source_definition(self) -> CompactDefinitionSource:
        self.require_closed()
        return self.creation.parent_context, self.creation.definition

    def proves_same_object(self, other: CapturedReferenceResolution) -> bool:
        if not isinstance(other, SourceDefinitionCapture):
            return False
        self.require_closed()
        other.require_closed()
        return (
            self.creation.execution is other.creation.execution
            and self.creation.node is other.creation.node
        )

    def require_release_from(
        self,
        kernel: CapturedReferenceKernel,
        slot: CapturedSlotQuery,
    ) -> None:
        if kernel is not self.creation.execution.kernel:
            raise ValueError("Source definition belongs to another execution admission")
        slot.require_independent_reference(kernel, self)


class SourceFunctionCreationABC(
    NativeTypeCapture,
    NativeDescriptorArgumentABC,
    OpaqueCapturedObjectOperations,
    SourceNativeFrameResolver,
):
    """Raw function birth at an original source cut, without a body activation."""

    violation = CapturedReferenceViolation.UNPROVED_ACCESS
    native_type: ClassVar[type] = FunctionType

    @property
    @abstractmethod
    def native_execution(self) -> NativeFunctionExecution:
        raise NotImplementedError

    @property
    @abstractmethod
    def creation_position(self) -> CompactFlowPosition:
        raise NotImplementedError

    def require_descriptor_argument(self) -> None:
        native = self.native_execution
        self.execution.module.native_compilation.execution_outcome.require_function(
            native
        )
        creation = native.require_creation()
        creation.frame.resolve(self)
        position = self.creation_position
        if position.branch_path or position.evaluation_path:
            raise ValueError(
                "Conditional or repeated function creation remains unproved"
            )
        _ = self.native_frame_prefix

    def require_native_function_operand(self, value: NativeFunctionValue) -> None:
        self.require_descriptor_argument()
        if (
            value.creation.instruction_offset
            != self.native_execution.require_creation().instruction_offset
        ):
            raise ValueError(
                "Native operand has a different original function creation"
            )


@dataclass(frozen=True, eq=False)
class SourceCreatedFunctionCapture(
    SourceDefinitionEntry[ast.FunctionDef | ast.AsyncFunctionDef],
    SourceDefinitionCapture,
    SourceFunctionCreationABC,
):
    """An installed plain function from one admitted source definition event.

    The native receipt proves raw creation, not body activation. Prefix and
    storage proof close the header and final installation independently. No
    analyzer function object, callable behavior or release guarantee is invented.
    """

    creation_position = AliasProperty[CompactFlowPosition]("definition.position")

    @classmethod
    def from_binding(
        cls, execution: SourceExecutionABC, binding: CompactMutation
    ) -> SourceCreatedFunctionCapture:
        node = cast(
            ast.FunctionDef | ast.AsyncFunctionDef,
            execution.source.event_operation(binding).node,
        )
        return cls(execution, node)

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_native_installation(prefix)
        self.require_complete_source_cut(prefix)
        return self.execution.module.native_compilation.return_after(
            self.native_execution
        )

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        """Join the original native store to its admitted source definition."""
        self.require_available_in(prefix)
        return self.require_installation_name(
            self.native_execution.require_installation()
        )

    @property
    def declaration(self) -> CompactFunctionDeclaration:
        self.require_original_operation()
        declaration = self.context.flow.owner.declaration
        if declaration is None:
            raise ValueError("Function result requires its actual function declaration")
        return declaration

    @property
    def native_execution(self) -> NativeFunctionExecution:
        """Rejoin original source and compiler declarations on every query."""
        declaration = self.declaration
        native = self.execution.module.native_compilation.execution_for(
            SourceByteSpan.require_node(self.node)
        )
        if declaration.execution is not native:
            raise ValueError(
                "Function creation remains unproved: different compiler receipt"
            )
        return native

    def call_authority(
        self,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CallAuthority:
        return SourceFunctionCall.for_call(environment, context, call)

    def definition_application_activation(
        self, application: SourceDefinitionDecoratorApplicationABC
    ) -> SourceFunctionActivationABC:
        return SourceDefinitionFunctionActivation(application, self)

    def require_definition_application_argument(self) -> None:
        self.require_descriptor_argument()

    def require_fresh_function_namespace(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        """Transport empty native birth storage; callable behavior remains unproved."""
        self.require_available_in(prefix)
        self.execution.entry.require_external_noninterference(prefix)
        self.execution.module.native_compilation.require_fresh_function_namespace(
            self.native_execution
        )

    def require_nonabstract_member(self, prefix: AdmittedExecutionPrefixABC) -> None:
        self.require_fresh_function_namespace(prefix)
        NativeCreationBackend.current().require_nonabstract_member_type(
            self.native_type
        )

    @property
    def creation(self) -> SourceDefinitionEntry:
        return self

    def result(self) -> CapturedReferenceResolution:
        result = self.creation_results[-1]
        result.require_closed()
        return result

    @cached_property
    def creation_results(
        self,
    ) -> tuple[SourceCreatedFunctionCapture | SourceNativeDecoratorApplication, ...]:
        return (
            self,
            *(
                SourceNativeDecoratorApplication(self, index)
                for index in range(len(self.definition.target.decorator_uses))
            ),
        )

    def require_closed(self) -> None:
        self.require_descriptor_argument()
        if self.definition.target.decorator_uses:
            raise ValueError("Function decorator result remains unproved")
        self.execution.require_binding_write(self.node)


class SourceFunctionActivationABC(ABC):
    """One exact source-function activation, independent of invocation syntax."""

    environment: NativeReferenceEnvironment
    callee: SourceCreatedFunctionCapture
    caller_prefix: AdmittedExecutionPrefixABC

    @abstractmethod
    def argument_values(
        self, parameter_name: str
    ) -> tuple[CapturedReferenceResolution, ...] | None:
        """Return values bound to one declaration-owned parameter."""
        raise NotImplementedError

    @abstractmethod
    def require_argument_value(self, value: CapturedReferenceResolution) -> None:
        """Discharge the value under this invocation family's argument protocol."""
        raise NotImplementedError

    @abstractmethod
    def activation_prefix(
        self, child: AdmittedExecutionPrefixABC
    ) -> AdmittedExecutionPrefixABC:
        """Join the actual invocation cut to a fresh callee interval."""
        raise NotImplementedError

    @abstractmethod
    def require_canonical_activation(self) -> None:
        """Require this activation to be the declaration owner's canonical edge."""
        raise NotImplementedError

    @cached_property
    def initial_entries(self) -> Mapping[NativeScalar, CapturedReferenceResolution]:
        """Bind exact entry values through the callee's source declaration."""
        parameters = FunctionParameterSource.from_arguments(self.callee.node.args)
        declaration = self.callee.declaration
        if len(parameters) != len(declaration.signature.parameters) or any(
            source.argument.arg != parameter.name or source.kind is not parameter.kind
            for source, parameter in zip(
                parameters, declaration.signature.parameters, strict=True
            )
        ):
            raise ValueError("Function entry parameters differ from their declaration")
        entries: dict[NativeScalar, CapturedReferenceResolution] = {}
        for source in parameters:
            values = self.argument_values(source.argument.arg)
            if source.kind.variadic:
                raise ValueError("Variadic function activation remains unproved")
            if values is None:
                if source.default is None:
                    raise ValueError("Function entry has no bound parameter value")
                value = self.callee.execution.capture_value(source.default)
                value.require_closed()
            else:
                if len(values) != 1:
                    raise ValueError(
                        "Function parameter requires one original argument"
                    )
                (value,) = values
                self.require_argument_value(value)
            entries[source.argument.arg] = value
        return MappingProxyType(entries)

    @property
    def entry_continuation(self) -> NativeReturn:
        """Rejoin exact binding and callee code before admitting its body walk."""
        _ = self.initial_entries
        function = self.callee
        function.native_execution.mode.require_immediate_activation()
        return function.execution.module.native_compilation.return_from(
            function.native_execution
        ).require_from_entry()

    @cached_property
    def activation(self) -> SourceFunctionExecution:
        return SourceFunctionExecution(SourceFunctionEntry(self))


class SourceFunctionCall(SignatureCallAuthorityABC, SourceFunctionActivationABC):
    """An original source-function invocation, not an assumed body activation."""

    @cached_property
    def callee(self) -> SourceCreatedFunctionCapture:
        value = super().callee
        if not isinstance(value, SourceCreatedFunctionCapture):
            raise TypeError("Source call requires its original function capture")
        return value

    @property
    def caller_prefix(self) -> AdmittedExecutionPrefixABC:
        function = self.callee
        prefix = function.require_available_at(
            self.environment.kernel, self.context, self.call.position
        )
        function.execution.entry.require_external_noninterference(prefix)
        return prefix

    @property
    def signature(self) -> CompactFunctionSignature:
        _ = self.caller_prefix
        function = self.callee
        return function.declaration.signature

    def argument_values(
        self, parameter_name: str
    ) -> tuple[CapturedReferenceResolution, ...] | None:
        argument = self.bound_arguments.argument_for(parameter_name)
        if argument is None:
            return None
        return tuple(
            self.environment.kernel._read_use(value, self.context, frozenset())
            for value in argument.values
        )

    def require_argument_value(self, value: CapturedReferenceResolution) -> None:
        value.require_closed()

    def activation_prefix(
        self, child: AdmittedExecutionPrefixABC
    ) -> AdmittedExecutionPrefixABC:
        return FunctionInvocationPrefix(
            self.caller_prefix,
            self.call,
            self.callee.declaration,
            child,
        )

    def require_canonical_activation(self) -> None:
        if self.environment.call_authority(self.context, self.call) is not self:
            raise ValueError("Function entry requires its canonical invocation")

    def require_closed(self) -> None:
        self.activation.require_closed()

    def result(self) -> CapturedReferenceResolution:
        return self.activation.result()


class SourceDefinitionDecoratorApplicationABC(
    SourceDefinitionResultABC,
    SourceNativeOperandInventoryABC,
    SourceDefinitionApplicationAuthorityABC,
    ABC,
):
    """Shared source/compiler topology for one original decorator application."""

    creation: SourceDefinitionEntry
    index: int

    execution = AliasProperty["SourceExecutionABC"]("creation.execution")
    environment = AliasProperty[NativeReferenceEnvironment]("creation.execution")
    operation = AliasProperty[SourceFlowOperation]("creation.operation")
    native_frame_context = AliasProperty[CompactFlowContext]("creation.parent_context")
    native_frame_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "creation.parent_prefix"
    )
    source_completion_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "creation.parent_prefix"
    )
    application_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "creation.parent_prefix"
    )
    definition_application = AliasProperty[CompactMutation[CompactDefinitionTarget]](
        "creation.definition"
    )

    @property
    def creation_results(self) -> tuple[SourceDefinitionResultABC, ...]:
        return self.creation.creation_results

    @property
    def production(self) -> NativeValueStore:
        return self.creation.native_result_store

    @property
    @abstractmethod
    def _native_application_path(
        self,
    ) -> tuple[NativeProducedValue, tuple[NativeCallValue, ...]]:
        """Derive raw production and applications from their compiler owner."""
        raise NotImplementedError

    @abstractmethod
    def _require_native_application_predecessors(
        self,
        raw: NativeProducedValue,
        applications: tuple[NativeCallValue, ...],
    ) -> None:
        """Validate the chain through the native authority that owns it."""
        raise NotImplementedError

    @property
    def native_value(self) -> NativeCallValue:
        self.require_original_application()
        return self._native_application_path[1][self.index]

    @property
    def decorator_use(self) -> CompactValueUse:
        return self.creation.definition.target.decorator_uses[-1 - self.index]

    operand = AliasProperty[CompactValueUse]("decorator_use")

    @property
    def argument(self) -> SourceDefinitionResultABC:
        return self.creation_results[self.index]

    @property
    def callee(self) -> CapturedReferenceResolution:
        native_value = self.native_value
        context = self.creation.parent_context
        self.execution.source_operation(context, self.decorator_use)
        return SourceNativeOperandJoin(
            self, CompactFlowValue(context, self.decorator_use)
        ).require_join(native_value.callee)

    def function_activation(self) -> SourceFunctionActivationABC:
        return self.callee.definition_application_activation(self)

    def native_definition_application_authority(
        self, callee: CapturedNativeObject
    ) -> NativeDefinitionApplicationAuthorityABC:
        del callee
        raise ValueError("Native definition application semantics remain unproved")

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        self.require_available_in(prefix)
        if self is not self.creation.creation_results[-1]:
            raise ValueError("Only the final definition result has an installation")
        return self.require_installation_name(self.production.binding)

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_native_installation(prefix)
        self.require_complete_source_cut(prefix)
        return self.production.require_return()

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.production.require_value(value)

    def require_original_application(self) -> None:
        results = self.creation_results
        decorators = self.creation.decorator_nodes
        if (
            not 0 <= self.index < len(decorators)
            or len(results) != len(decorators) + 1
            or results[self.index + 1] is not self
        ):
            raise ValueError(
                "Decorator application requires its original creation chain"
            )
        raw, applications = self._native_application_path
        if len(applications) != len(decorators):
            raise ValueError(
                "Native applications differ from the original source chain"
            )
        for application, decorator in zip(
            applications, reversed(decorators), strict=True
        ):
            if application.source_span != SourceByteSpan.require_node(decorator):
                raise ValueError(
                    "Native application differs from its original decorator chain"
                )
        self._require_native_application_predecessors(raw, applications)


@dataclass(frozen=True, eq=False)
class SourceNativeDecoratorApplication(
    NativeDescriptorResult,
    SourceDefinitionDecoratorApplicationABC,
):
    """Native descriptor application over the shared definition topology."""

    creation: SourceCreatedFunctionCapture
    index: int

    @property
    def _native_application_path(
        self,
    ) -> tuple[NativeProducedValue, tuple[NativeCallValue, ...]]:
        native = self.creation.native_execution
        installed = native.require_applied_installation()
        store = self.production
        if (
            store.binding is not installed
            or store.frame is not native.require_creation().frame
        ):
            raise ValueError(
                "Decorator store differs from its original native installation"
            )
        raw = store.production_at(native.require_creation().instruction_offset)
        applications = tuple(
            application.operand_in(store)
            for application in native.require_applications()
        )
        return raw, applications

    def _require_native_application_predecessors(
        self,
        raw: NativeProducedValue,
        applications: tuple[NativeCallValue, ...],
    ) -> None:
        native = self.creation.native_execution
        creation = native.require_creation()
        creation.require_definition_operand(raw)
        preceding: NativeCaptureSite | NativeDefinitionApplication = creation
        for receipt in native.require_applications():
            if receipt.argument is not preceding:
                raise ValueError(
                    "Native application differs from its compiler predecessor chain"
                )
            preceding = receipt

    @property
    def descriptor_argument(self) -> NativeDescriptorArgumentABC:
        return cast(NativeDescriptorArgumentABC, self.argument)

    def require_closed(self) -> None:
        _ = self.native_value
        self.descriptor_argument.require_descriptor_argument()
        try:
            self.callee.require_native(self.native_declarations)
        except ValueError as error:
            raise ValueError("Function decorator result remains unproved") from error
        self.execution.require_binding_write(self.creation.node)


@dataclass(frozen=True, eq=False)
class SourceDefinitionFunctionActivation(SourceFunctionActivationABC):
    """A source function invoked by one exact compiler definition application."""

    application: SourceDefinitionDecoratorApplicationABC
    function: SourceCreatedFunctionCapture

    environment = AliasProperty[NativeReferenceEnvironment]("application.execution")

    @property
    def callee(self) -> SourceCreatedFunctionCapture:
        if not self.application.callee.proves_same_object(self.function):
            raise ValueError(
                "Definition application activation has a different source callee"
            )
        return self.function

    @property
    def caller_prefix(self) -> AdmittedExecutionPrefixABC:
        self.application.require_original_application()
        prefix = self.application.application_prefix
        self.callee.execution.entry.require_external_noninterference(prefix)
        return prefix

    @cached_property
    def bound_arguments(
        self,
    ) -> CompactCallBinding[CapturedReferenceResolution]:
        binding = self.callee.declaration.signature.bind(
            (CompactCallArgument(self.application.argument),), ()
        )
        if not binding.is_exact:
            raise ValueError(f"Argument binding rejected: {binding.violation}")
        return binding

    def argument_values(
        self, parameter_name: str
    ) -> tuple[CapturedReferenceResolution, ...] | None:
        argument = self.bound_arguments.argument_for(parameter_name)
        return None if argument is None else argument.values

    def require_argument_value(self, value: CapturedReferenceResolution) -> None:
        if value is not self.application.argument:
            raise ValueError(
                "Definition application binding has a foreign implicit argument"
            )
        value.require_definition_application_argument()

    def activation_prefix(
        self, child: AdmittedExecutionPrefixABC
    ) -> AdmittedExecutionPrefixABC:
        _ = self.caller_prefix
        return DefinitionApplicationPrefix(
            self.application,
            self.callee.declaration,
            child,
        )

    def require_canonical_activation(self) -> None:
        self.application.require_original_application()
        _ = self.callee


@dataclass(frozen=True, eq=False)
class SourceCreatedClassCapture(
    SourceDefinitionCapture,
    NativeTypeCapture,
    InstanceProtocolEvidenceABC,
    OpaqueCapturedObjectOperations,
    SourceNativeOperandInventoryABC,
):
    """Closed source class creation without inventing a native runtime object."""

    entry: SourceClassEntry
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    native_type: ClassVar[type] = type

    creation = AliasProperty[SourceDefinitionEntry]("entry")
    execution = AliasProperty["SourceExecutionABC"]("entry.execution")
    operation = AliasProperty[SourceFlowOperation]("entry.operation")
    native_frame_context = AliasProperty[CompactFlowContext]("entry.parent_context")
    native_frame_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "entry.parent_prefix"
    )
    source_completion_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "entry.parent_prefix"
    )

    production = AliasProperty[NativeValueStore]("entry.native_result_store")

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.production.require_value(value)

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        self.require_available_in(prefix)
        if self.entry.definition.target.decorator_uses:
            raise ValueError(
                "Only the final class definition result has an installation"
            )
        _ = self.native_construction
        return self.require_installation_name(self.production.binding)

    @property
    def native_construction(self) -> NativeCallValue:
        """Join raw class construction without claiming the transformed installation."""
        store = self.production
        value = self.entry.capture.definition_construction_in(store)
        _ = self.entry.builder_value
        if (
            value.arguments[1].require_scalar_store_value().require_native_text()
            != self.entry.node.name
        ):
            raise ValueError("Native class call differs from its original source name")
        value.require_argument_shape(
            len(self.entry.node.bases) + 2,
            tuple(keyword.arg for keyword in self.entry.node.keywords),
        )
        for argument, operand in zip(
            (
                *self.entry.node.bases,
                *(keyword.value for keyword in self.entry.node.keywords),
            ),
            value.arguments[2:],
            strict=True,
        ):
            SourceNativeOperandJoin(
                self, self.execution.source.value_reads_by_node[argument]
            ).require_join(operand)
        return value

    def require_definition_application_argument(self) -> None:
        _ = self.entry.completed
        _ = self.native_construction

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_native_installation(prefix)
        self.require_complete_source_cut(prefix)
        return self.production.require_return()

    def require_inert_instance_hooks(
        self,
        hooks: tuple[NativeDeclaration, ...],
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> None:
        self.require_available_at(resolver, context, position)
        self.entry.require_absent_native_hooks(hooks)

    def call_authority(
        self,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CallAuthority:
        return SourceObjectConstruction.for_call(environment, context, call)

    def require_closed(self) -> None:
        self.entry.execution.require_class_creation(self.entry.node)

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        self.require_available_at(resolver, context, position)
        _ = self.entry.plain_subclass_protocol
        return self.entry.mro_type


class SourceObjectConstruction(DefaultObjectConstruction):
    """One default object construction; no analyzer instance substitutes for it."""

    def require_instance_hooks(self, hooks: tuple[NativeDeclaration, ...]) -> None:
        self.constructor.require_inert_instance_hooks(
            hooks,
            self.environment.kernel,
            self.context,
            self.call.position,
        )

    @cached_property
    def constructor(self) -> InstanceProtocolEvidenceABC:
        if not isinstance(self.callee, InstanceProtocolEvidenceABC):
            raise ValueError("Callee has no instance protocol evidence")
        return self.callee


class SourceNativeStorageABC(
    SourceNativeProductionABC,
    NativeCompletionResolverABC,
    NativeValueResolverABC[CapturedReferenceResolution],
    NativeBindingTransferResolverABC[tuple[str, CapturedReferenceResolution] | None],
):
    """Interpret original native operands and writes at one admitted source cut."""

    execution: SourceExecutionABC

    def _native_value_completion(self, value: NativeProducedValue) -> None:
        self.native_value(value).require_closed()

    def _native_stack_effect_completion(self, effect: NativeStackEffectABC) -> None:
        self._require_native_production(effect)
        raise ValueError("Native stack effect completion remains unproved")

    def _local_native_value_resolution(
        self, value: NativeLocalValue
    ) -> CapturedReferenceResolution:
        raise ValueError("Native fast-local read at this source cut remains unproved")

    def _global_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        raise ValueError("Native global store remains unproved")

    def _local_ensure_resolution(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        previous = self._preceding_native_local_value(
            binding.name, binding.instruction_offset
        )
        return (
            self._local_store_resolution(binding)
            if previous is None
            else (binding.name, previous)
        )

    def _local_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        if binding.value is None:
            raise ValueError("Native local store has no proved operand")
        previous = self._preceding_native_local_value(
            binding.name, binding.instruction_offset
        )
        if previous is not None:
            previous.require_release_in(self.native_lookup_prefix.endpoint.frame)
        return binding.name, self.native_value(binding.value)

    def native_value(self, value: NativeProducedValue) -> CapturedReferenceResolution:
        self.require_native_value(value)
        return value.resolve(self)

    def _unproved_native_value_resolution(
        self, value: NativeProducedValue
    ) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)

    def _typed_native_value_resolution(
        self, value: NativeTypedValue
    ) -> CapturedReferenceResolution:
        return SourceNativeTypeCapture(self, value)

    def _preceding_native_local_value(
        self, name: NativeScalar, offset: int
    ) -> CapturedReferenceResolution | None:
        for binding in reversed(self.native_bindings):
            if binding.name == name and binding.instruction_offset < offset:
                result = binding.resolve(self)
                if result is not None:
                    return result[1]
        return self._native_initial_local(name)

    def _name_native_value_resolution(
        self, value: NativeNameValue
    ) -> CapturedReferenceResolution:
        local = self._preceding_native_local_value(value.name, value.instruction_offset)
        return self._global_native_value_resolution(value) if local is None else local

    def _global_native_value_resolution(
        self, value: NativeReadValue
    ) -> CapturedReferenceResolution:
        return self.execution.kernel._namespaces_resolution(
            self.native_lookup_prefix,
            self.native_global_namespaces,
            value.name,
            frozenset(),
        )

    @property
    @abstractmethod
    def native_lookup_prefix(self) -> AdmittedExecutionPrefixABC:
        raise NotImplementedError

    @property
    @abstractmethod
    def native_bindings(self) -> tuple[NativeBindingTransfer, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def native_global_namespaces(
        self,
    ) -> tuple[NamespaceEvidenceABC | OpenCapturedReference, ...]:
        raise NotImplementedError

    @abstractmethod
    def _native_initial_local(
        self, name: NativeScalar
    ) -> CapturedReferenceResolution | None:
        raise NotImplementedError

    def require_native_value(self, value: NativeProducedValue) -> None:
        self._require_native_production(value)
        for dependency in value.inputs:
            self.native_value(dependency).require_closed()

    def _cell_store_resolution(self, binding: NativeBindingTransfer) -> None:
        raise ValueError("Native cell store at this source cut remains unproved")

    def _cell_creation_resolution(self, binding: NativeBindingTransfer) -> None:
        raise ValueError("Native cell creation at this source cut remains unproved")


class SourceNativePrefixStorageABC(SourceNativeStorageABC):
    """Native reads use the namespaces and position of their admitted source cut."""

    def _fast_local_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        return self._local_store_resolution(binding)

    def _local_native_value_resolution(
        self, value: NativeLocalValue
    ) -> CapturedReferenceResolution:
        local = self._preceding_native_local_value(value.name, value.instruction_offset)
        if local is None:
            raise ValueError("Native fast local is absent at the original source cut")
        return local

    def _native_initial_local(
        self, name: NativeScalar
    ) -> CapturedReferenceResolution | None:
        endpoint = self.native_lookup_prefix.endpoint
        return self.execution.kernel._slot(
            endpoint.frame.locals,
            name,
            endpoint.context,
            endpoint.position,
            frozenset(),
        )

    @property
    def native_global_namespaces(
        self,
    ) -> tuple[NamespaceEvidenceABC | OpenCapturedReference, ...]:
        frame = self.native_lookup_prefix.endpoint.frame
        return frame.globals, frame.builtins


class SourceNativeNamespaceABC(SourceNativeStorageABC):
    """Class-native storage additionally owns the original compiler cells."""

    def _annotation_namespace_native_value_resolution(
        self, value: NativeAnnotationNamespaceValue
    ) -> CapturedReferenceResolution:
        namespace = self.native_class_entry.annotation_namespace
        if namespace.value is not value:
            raise ValueError("Annotation namespace belongs to another original setup")
        namespace.require_admitted(self.execution.initial)
        return namespace

    def _local_native_value_resolution(
        self, value: NativeLocalValue
    ) -> CapturedReferenceResolution:
        self.native_class_entry.capture.prologue.require_fresh_cell(
            value.name, value.operand_index, value.instruction_offset
        )
        return SourceFreshCellCapture(self, value)

    @property
    @abstractmethod
    def native_class_entry(self) -> SourceClassBodyEntryABC:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class SourceNativeCaptureABC(NativeTypeCapture, OpaqueCapturedObjectOperations):
    """Source-bound value proof retaining its original native production."""

    entry: SourceNativeStorageABC
    value: NativeProducedValue
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    def require_closed(self) -> None:
        self.entry.native_value(self.value)


@dataclass(frozen=True, eq=False)
class SourceNativeTypeCapture(SourceNativeCaptureABC):
    """An intrinsic native production type, independent of frame lookup."""

    value: NativeTypedValue
    native_type = AliasProperty[type]("value.native_type")

    def _native_scalar_value(self) -> object:
        return self.value.require_native_scalar()


@dataclass(frozen=True, eq=False)
class SourceFreshCellCapture(SourceNativeCaptureABC):
    """Original compiler cell and its unmodified construction-field installation."""

    value: NativeLocalValue
    native_type: ClassVar[type] = CellType

    entry: SourceNativeNamespaceABC

    def require_class_construction_field(
        self,
        field: CPythonClassConstructionField,
        namespace: NamespaceCreationEvidenceABC,
    ) -> None:
        self.require_closed()
        if self.entry.native_class_entry is not namespace:
            raise ValueError("Construction cell belongs to a foreign native frame")
        bindings = tuple(
            binding
            for binding in self.entry.native_bindings
            if binding.name == field.value and binding.value is self.value
        )
        if len(bindings) != 1:
            field.require_default()


@dataclass(frozen=True, eq=False)
class SourceAnnotationNamespace(EmptyDictionaryCreation):
    """Fresh class-entry annotation storage; ordinary item mutations own its contents."""

    entry: SourceClassBodyEntryABC
    value: NativeAnnotationNamespaceValue
    initial = AliasProperty[InitialNativeIsland]("entry.execution.initial")

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if initial is not self.initial or self.entry.annotation_namespace is not self:
            raise ValueError(
                "Annotation namespace requires its original class activation"
            )
        self.entry.require_preparation()
        prologue = self.entry.capture.prologue
        prologue.require_value(self.value)
        bindings = tuple(
            binding
            for binding in prologue.bindings
            if binding.name == self.value.namespace_name
            and binding.instruction_offset <= self.value.instruction_offset
        )
        if len(bindings) != 1 or bindings[0].value is not self.value:
            raise ValueError("Fresh annotation namespace requires an absent entry key")

    def require_available(
        self, kernel: CapturedReferenceKernel, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        self.require_admitted(kernel.initial)
        if kernel is not self.entry.execution.kernel:
            raise ValueError("Annotation namespace belongs to another execution kernel")
        if not any(
            interval.frame.locals is self.entry for interval in prefix.intervals
        ):
            prefix.require_event(
                self.entry.parent_context,
                self.entry.definition,
                self.entry.parent_prefix.endpoint.frame,
            )


@dataclass(frozen=True, eq=False)
class SourceModuleAnnotationNamespace(EmptyDictionaryCreation):
    """Actual module SETUP_ANNOTATIONS creation after an absent entry key."""

    execution: SourceModuleExecution
    initial = AliasProperty[InitialNativeIsland]("execution.initial")

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        contents = self.execution.initial_contents
        contents.require_closed()
        if initial is not self.initial or contents.annotation_namespace is not self:
            raise ValueError("Module annotation storage belongs to another activation")
        setup = contents.setup
        if setup is None or self.execution.entry.member(setup.binding.name) is not None:
            raise ValueError("Fresh module annotation storage requires initial absence")

    def require_available(
        self, kernel: CapturedReferenceKernel, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        self.require_admitted(kernel.initial)
        if kernel is not self.execution.kernel:
            raise ValueError("Module annotation storage belongs to another kernel")
        self.execution.entry.require_available(kernel, prefix)


@dataclass(frozen=True)
class SourceModuleEntryContents(InitialNamespaceContents):
    """Project original native entry work without changing supplied entry facts."""

    execution: SourceModuleExecution
    kernel = AliasProperty[CapturedReferenceKernel]("execution.kernel")
    namespace = AliasProperty[NamespaceEvidenceABC]("execution.entry")

    @cached_property
    def setup(self) -> NativeValueStore | None:
        return self.execution.module.native_compilation.module_annotation_setup

    def require_closed(self) -> None:
        super().require_closed()
        _ = self.setup

    @cached_property
    def annotation_namespace(self) -> SourceModuleAnnotationNamespace:
        return SourceModuleAnnotationNamespace(self.execution)

    @property
    def names(self) -> frozenset[NativeScalar]:
        names = super().names
        return names if self.setup is None else names | {self.setup.binding.name}

    def member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        value = super().member(key)
        if value is None and self.setup is not None and key == self.setup.binding.name:
            return self.annotation_namespace
        return value


@dataclass(frozen=True, eq=False)
class SourceEmptyDictionaryCapture(EmptyDictionaryCreation):
    """One actual empty dictionary production in an admitted source activation.

    The namespace is its creation receipt, never an analyzer-allocated dictionary.
    Initial absence does not interpret or admit subsequent item stores.
    """

    execution: SourceExecutionABC
    operation: SourceFlowOperation
    initial = AliasProperty[InitialNativeIsland]("execution.initial")

    @property
    def context(self) -> CompactFlowContext:
        return self.execution.context_for_owner(self.operation.owner)

    @property
    def use(self) -> CompactValueUse:
        event = self.operation.event
        if not isinstance(event, CompactValueUse):
            raise ValueError("Empty dictionary requires its original value production")
        return event

    @cached_property
    def creation_admission(self) -> None:
        if self.use.position.branch_path or self.use.position.evaluation_path:
            raise ValueError(
                "Conditional or repeated empty dictionary creation remains unproved"
            )
        self.execution.required_prefix(self.context, self.use.position)
        NativeCreationBackend.current().require_empty_dictionary_creation(
            cast(ast.Dict, self.operation.node)
        )

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if (
            initial is not self.initial
            or self.execution.empty_dictionary_creation(
                CompactFlowValue(self.context, self.use)
            )
            is not self
        ):
            raise ValueError("Empty dictionary has no canonical source creation")
        _ = self.creation_admission

    def require_available(
        self,
        kernel: CapturedReferenceKernel,
        prefix: AdmittedExecutionPrefixABC,
    ) -> None:
        self.require_admitted(kernel.initial)
        if kernel is not self.execution.kernel:
            raise ValueError("Empty dictionary belongs to another execution kernel")
        prefix.require_event(
            self.context,
            self.use,
            self.execution.required_prefix(
                self.context, self.use.position
            ).endpoint.frame,
        )


@dataclass(frozen=True, eq=False)
class SourceTupleCapture(NativeTypeCapture, OpaqueCapturedObjectOperations):
    """Ordered retained values at an original tuple production, not native identity.

    The existing kernel owns query reuse. Elements derive from their original
    cuts, including nested productions. Release deliberately retains the
    native type contract's refusal for content-bearing objects.
    """

    execution: SourceExecutionABC
    read: CompactFlowValue
    pending: frozenset[CompactBindingVisit[CompactFlowContext]]
    native_type: ClassVar[type] = tuple
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    def _native_constant_value(self) -> object:
        self.require_closed()
        return tuple(value._native_constant_value() for value in self.elements)

    @property
    def production(self) -> CompactTupleValue:
        operation = self.execution.source.value_operation(self.read)
        use = self.read.use
        if not isinstance(use, CompactValueUse) or not isinstance(
            use.value, CompactTupleValue
        ):
            raise ValueError("Tuple capture requires its original compact production")
        if use.position.branch_path or use.position.evaluation_path:
            raise ValueError("Conditional or repeated tuple creation remains unproved")
        node = operation.node
        NativeCreationBackend.current().require_tuple_construction(node)
        if len(use.value.inputs) != len(node.elts):
            raise ValueError("Tuple production has a different input count")
        previous = None
        for element, input_use in zip(node.elts, use.value.inputs, strict=True):
            original = self.execution.source.value_operation(
                CompactFlowValue(self.read.context, input_use)
            )
            if original.node is not element:
                raise ValueError("Tuple input is not the original element production")
            if not input_use.position.dominates(use.position) or (
                previous is not None and not previous.dominates(input_use.position)
            ):
                raise ValueError("Tuple inputs have no original ordered completion")
            previous = input_use.position
        return use.value

    @cached_property
    def elements(self) -> tuple[CapturedReferenceResolution, ...]:
        values = tuple(
            self.execution.kernel._read_use(input_use, self.read.context, self.pending)
            for input_use in self.production.inputs
        )
        for value in values:
            value.require_closed()
        return values

    def require_closed(self) -> None:
        _ = self.production
        self.execution.required_prefix(self.read.context, self.read.use.position)
        _ = self.elements


@dataclass(frozen=True, eq=False)
class SourceReadFrame(SourceNativeFrameResolver):
    """One original read and its admitted executing frame."""

    execution: SourceExecutionABC
    read: CompactFlowValue
    native_frame_context = AliasProperty[CompactFlowContext]("read.context")

    @cached_property
    def native_frame_prefix(self) -> AdmittedExecutionPrefixABC:
        self.execution.source.value_operation(self.read)
        return self.execution.required_prefix(self.read.context, self.read.use.position)


class SourceReadCaptureABC(
    NativeTypeCapture,
    OpaqueCapturedObjectOperations,
    SourceReadFrame,
):
    """A captured expression retains its original read and executing frame."""

    violation = CapturedReferenceViolation.UNPROVED_ACCESS


@dataclass(frozen=True, eq=False)
class SourceLiteralCapture(SourceReadCaptureABC):
    """Known literal behavior from one original evaluation, not its object identity.

    Type and immutable scalar contents derive from the same original literal.
    The analyzer's parsed value never supplies runtime-object identity.
    Release follows the native exact-type contract; other protocols stay open.
    """

    def _native_scalar_value(self) -> object:
        return self.literal_value

    @cached_property
    def literal_value(self) -> object:
        operation = self.execution.source.value_operation(self.read)
        if not isinstance(
            cast(CompactValueUse, self.read.use).value, OpaqueValueExpression
        ):
            raise ValueError(
                "Literal capture requires an original expression value read"
            )
        return LiteralExpressionEffects(cast(ast.expr, operation.node)).value

    @cached_property
    def native_type(self) -> type:
        return type(self.literal_value)

    def require_closed(self) -> None:
        _ = self.native_type
        _ = self.native_frame_prefix


@dataclass(frozen=True, eq=False)
class SourceFunctionExpressionCapture(SourceReadCaptureABC, SourceFunctionCreationABC):
    """Original expression creation, independent of a destination binding."""

    creation_position = AliasProperty[CompactFlowPosition]("read.use.position")

    @property
    def native_execution(self) -> NativeFunctionExecution:
        operation = self.execution.source.value_operation(self.read)
        if (
            not isinstance(self.read.use, CompactValueUse)
            or not isinstance(self.read.use.value, FunctionExpression)
            or not isinstance(operation.node, ast.Lambda)
        ):
            raise ValueError("Function creation requires its original expression read")
        return self.execution.module.native_compilation.execution_for(
            SourceByteSpan.require_node(operation.node)
        )

    def require_closed(self) -> None:
        self.require_descriptor_argument()


@dataclass(frozen=True, eq=False)
class SourceCompilerStoredCapture(
    NativeTypeCapture,
    OpaqueCapturedObjectOperations,
    SourceReadFrame,
):
    """Original compiler text at its source read, independently of runtime identity."""

    violation = CapturedReferenceViolation.UNPROVED_ACCESS
    native_type = AliasProperty[type]("production.value.native_type")

    def _native_scalar_value(self) -> object:
        return self.production.value.require_native_scalar()

    @cached_property
    def production(self) -> NativeConstantStore:
        operation = self.execution.source.value_operation(self.read)
        if not isinstance(
            cast(CompactValueUse, self.read.use).value, CompilerStoredValue
        ):
            raise ValueError("Compiler value requires its original marked source read")
        results = tuple(
            result
            for result in self.read.context.flow.evaluated_results
            if result.value_use is self.read.use
        )
        if len(results) != 1 or results[0].destination.direct_binding_name is None:
            raise ValueError("Compiler value requires one original bound result")
        self.execution.source_operation(self.read.context, results[0])
        production = self.execution.module.native_compilation.constant_store_for(
            SourceByteSpan.require_node(operation.node),
            results[0].destination.direct_binding_name,
        )
        production.frame.resolve(self)
        return production

    def require_closed(self) -> None:
        _ = self.production
        _ = self.native_frame_prefix


@dataclass(frozen=True, eq=False)
class SourceCompilerOperandRead(SourceReadFrame, SourceNativePrefixStorageABC):
    """Resolve an original implicit operand, with no manufactured source expression."""

    native_bindings: ClassVar[tuple[NativeBindingTransfer, ...]] = ()
    native_lookup_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "native_frame_prefix"
    )

    @property
    def operation(self) -> SourceFlowOperation:
        operation = self.execution.source.value_operation(self.read)
        if not isinstance(self.read.use, CompactCompilerOperandUse):
            raise ValueError("Compiler operand requires its original implicit read")
        return operation

    @property
    def receipt(self) -> NativeReturn:
        self.operation
        receipt = self.read.use.native_receipt(self.execution.source, self.read)
        receipt.frame.resolve(self)
        return receipt

    @property
    def operand(self) -> NativeProducedValue:
        self.receipt
        return self.read.use.native_operand(self.execution.source, self.read)

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.receipt.require_value(value)

    def result(self) -> CapturedReferenceResolution:
        value = self.native_value(self.operand)
        value.require_closed()
        return value


class SourceEventReturnABC(SourceCompletedReturnABC):
    """One original source event and its canonical executing frame."""

    execution: SourceExecutionABC

    source_completion_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "native_frame_prefix"
    )

    @property
    @abstractmethod
    def event(self) -> SourceFlowEvent:
        raise NotImplementedError

    @property
    def operation(self) -> SourceFlowOperation:
        return self.execution.source.event_operation(self.event)

    @property
    def native_frame_context(self) -> CompactFlowContext:
        return self.execution.context_for_owner(self.operation.owner)

    @property
    def native_frame_prefix(self) -> AdmittedExecutionPrefixABC:
        return self.execution.required_prefix(
            self.native_frame_context, self.event.position
        )


@dataclass(frozen=True, eq=False)
class SourceBindingReturnABC(SourceEventReturnABC, SourceInstalledReturnABC):
    """A canonical source binding and its original executing frame."""

    execution: SourceExecutionABC
    binding: CompactMutation
    event = AliasProperty[CompactMutation]("binding")


class SourceImportStoreABC(SourceBindingReturnABC):
    """Original import request, its admitted source binding and native transfer."""

    @property
    def origin(self) -> ImportedNameOrigin:
        self.operation
        if not isinstance(self.binding.target, CompactImportTarget):
            raise ValueError("Import installation requires its original import target")
        return self.binding.target.origin

    @property
    def production(self) -> NativeValueStore:
        span = SourceByteSpan.require_node(self.operation.node)
        store = self.execution.module.native_compilation.value_store_for(
            span, span, self.origin.bound_name
        )
        store.frame.resolve(self)
        return store

    @abstractmethod
    def require_original_request(self, store: NativeValueStore) -> NativeImportValue:
        raise NotImplementedError

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        self.require_event_available_in(prefix, self.binding)
        self.execution.require_import_operation(self.operation)
        store = self.production
        self.require_original_request(store)
        return store.binding

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_complete_source_cut(prefix)
        self.native_completion_offset(prefix)
        return self.production.require_return()


class SourceModuleImportStore(SourceImportStoreABC):
    def require_original_request(self, store: NativeValueStore) -> NativeImportValue:
        value = store.value
        if not isinstance(value, NativeImportValue):
            raise ValueError("Module import has no original native request result")
        for operand in value.inputs:
            store.require_value(operand)
        value.require_request(self.origin.alias.name, 0, None)
        return value


class SourceMemberImportStore(SourceImportStoreABC):
    def require_original_request(self, store: NativeValueStore) -> NativeImportValue:
        value = store.value
        if (
            not isinstance(value, NativeImportMemberValue)
            or value.name != self.origin.alias.name
        ):
            raise ValueError("Member import differs from its original source alias")
        (module,) = value.inputs
        store.require_value(module)
        if not isinstance(module, NativeImportValue):
            raise ValueError("Member import has no original module request")
        for operand in module.inputs:
            store.require_value(operand)
        (reference,) = self.origin.declaration.module_references
        module.require_request(
            reference.module_name,
            reference.level,
            tuple(alias.name for alias in self.origin.declaration.aliases),
        )
        return module

    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        installed = self.require_native_installation(prefix)
        origin = self.origin
        if origin.alias_index != len(origin.declaration.aliases) - 1:
            raise ValueError("Import cleanup requires the final original alias store")
        store = self.production
        module = self.require_original_request(store)
        returned = store.require_return()
        release = returned.effect_for(
            SourceByteSpan.require_node(self.operation.node), NativeDiscardValue
        )
        if (
            release.instruction_offset <= installed.instruction_offset
            or release.inputs[0] is not module
        ):
            raise ValueError(
                "Import cleanup does not release its original module operand"
            )
        self.execution.initial.require_registered_module_retention(
            self.execution.kernel.member_import_module_name(
                origin, (self.native_frame_context, self.binding, frozenset())
            ),
            prefix,
        )
        return release.instruction_offset


class SourceDeletionReturn(SourceBindingReturnABC):
    """Join a source-proved deletion to its original native continuation.

    Presence, release and destination storage belong to the admitted source
    event. The native receipt authenticates the operation and executing frame;
    it does not replay that deletion against the resulting namespace.
    """

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_event_available_in(prefix, self.binding)
        self.require_complete_source_cut(prefix)
        name = self.binding.target.bound_name
        if name is None:
            raise ValueError("Native deletion requires a lexical binding")
        span = SourceByteSpan.require_node(self.operation.node)
        receipt = self.execution.module.native_compilation.return_after_binding(
            span, name
        )
        receipt.frame.resolve(self)
        receipt.binding_for(span, name).operation.require_deletion()
        return receipt

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        receipt = self.return_continuation(prefix)
        return receipt.binding_for(
            SourceByteSpan.require_node(self.operation.node),
            cast(str, self.binding.target.bound_name),
        )


class SourceNativeExpressionABC(
    SourceNativePrefixStorageABC, SourceNativeOperandInventoryABC
):
    """Join an original native production at its corresponding source read cut."""

    native_bindings: ClassVar[tuple[NativeBindingTransfer, ...]] = ()

    def _call_native_value_resolution(
        self, value: NativeCallValue
    ) -> CapturedReferenceResolution:
        node = self.source_read.source_operation(self.execution.source).node
        context, call = self.execution.source_call(node)
        result = self.execution.call_result(context, call)
        result.require_closed()
        callee = self.execution.source.reference_reads_by_node[node.func]
        SourceNativeOperandJoin(self, callee).require_join(value.callee)
        arguments = call.arguments
        if any(
            argument.is_unpacked
            for argument in (*arguments.positional, *arguments.keywords)
        ):
            raise ValueError("Native call does not prove unpacked arguments")
        value.require_argument_shape(
            len(arguments.positional),
            tuple(argument.name for argument in arguments.keywords),
        )
        for argument, operand in zip(arguments.values, value.arguments, strict=True):
            SourceNativeOperandJoin(
                self, CompactFlowValue(context, argument)
            ).require_join(operand)
        return result

    @property
    @abstractmethod
    def source_read(self) -> CompactFlowValue:
        raise NotImplementedError

    @property
    def source_value(self) -> CapturedReferenceResolution:
        self.source_read.source_operation(self.execution.source)
        return self.execution.kernel._read_use(
            self.source_read.use, self.source_read.context, frozenset()
        )

    def require_native_value(self, value: NativeProducedValue) -> None:
        self._require_native_production(value)
        for operand in value.graph_children:
            self._require_native_production(operand)
        self.source_read.use.require_native_origin(
            self.execution.source, self.source_read, value
        )
        self.source_value.require_closed()

    def require_join(self, value: NativeProducedValue) -> CapturedReferenceResolution:
        actual = self.native_value(value)
        source = self.source_value
        actual.require_closed()
        if actual is not source and not actual.proves_same_object(source):
            raise ValueError("Native store does not join the original source value")
        return source

    def _subscription_native_value_resolution(
        self, value: NativeSubscriptionValue
    ) -> CapturedReferenceResolution:
        source = self.execution.source
        node = self.source_read.source_operation(source).node
        operation = source.node_operation(node, CompactSubscription)
        authority = self.execution.subscription_authority(
            self.source_read.context, operation.event
        )
        authority.require_closed()
        for use, operand in zip(
            (authority.invocation.receiver_use, authority.invocation.argument_use),
            value.inputs,
            strict=True,
        ):
            SourceNativeOperandJoin(
                self, CompactFlowValue(self.source_read.context, use)
            ).require_join(operand)
        return self.source_value

    def _attribute_native_value_resolution(
        self, value: NativeAttributeValue
    ) -> CapturedReferenceResolution:
        source = self.execution.source
        node = self.source_read.source_operation(source).node
        if not isinstance(node, ast.Attribute) or node.attr != value.name:
            raise ValueError("Native attribute differs from its original source read")
        read = source.reference_reads_by_node.get(node.value)
        if read is None:
            raise ValueError("Native attribute has no original receiver read")
        (receiver,) = value.inputs
        SourceNativeOperandJoin(self, read).require_join(receiver)
        return self.source_value

    def _tuple_native_value_resolution(
        self, value: NativeTupleValue
    ) -> CapturedReferenceResolution:
        source = self.source_value
        if not isinstance(source, SourceTupleCapture):
            raise ValueError(
                "Native tuple requires an original source tuple construction"
            )
        inputs = source.production.inputs
        if len(inputs) != len(value.inputs):
            raise ValueError("Native tuple has different original source operands")
        for use, operand in zip(inputs, value.inputs, strict=True):
            SourceNativeOperandJoin(
                self, CompactFlowValue(self.source_read.context, use)
            ).require_join(operand)
        return source

    def _list_native_value_resolution(
        self, value: NativeListValue
    ) -> CapturedReferenceResolution:
        source = self.source_value
        if not isinstance(source, SourceLiteralCapture):
            raise ValueError("Native list requires its original literal construction")
        for operand in value.productions():
            self._require_native_production(operand)
        value.require_literal_contents(source.literal_value)
        return source

    def _empty_dictionary_native_value_resolution(
        self, value: NativeEmptyDictionaryValue
    ) -> CapturedReferenceResolution:
        capture = self.execution.empty_dictionary_creation(self.source_read)
        capture.require_available(self.execution.kernel, self.source_completion_prefix)
        return capture

    def _function_native_value_resolution(
        self,
        value: NativeFunctionValue,
    ) -> CapturedReferenceResolution:
        source = self.source_value
        if not isinstance(source, SourceFunctionCreationABC):
            raise ValueError(
                "Native function requires original source creation evidence"
            )
        source.require_native_function_operand(value)
        return source

    def _typed_native_value_resolution(
        self, value: NativeTypedValue
    ) -> CapturedReferenceResolution:
        source = self.source_value
        source.require_same_native_constant(value)
        return source

    @property
    def native_lookup_prefix(self) -> AdmittedExecutionPrefixABC:
        return self.execution.required_prefix(
            self.source_read.context, self.source_read.use.position
        )


@dataclass(frozen=True, eq=False)
class SourceNativeOperandJoin(SourceNativeExpressionABC):
    """One operand at its own read cut, under its original store inventory."""

    parent: SourceNativeOperandInventoryABC
    read: CompactFlowValue
    source_read = AliasProperty[CompactFlowValue]("read")
    execution = AliasProperty["SourceExecutionABC"]("parent.execution")

    source_completion_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "parent.source_completion_prefix"
    )

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.parent._require_native_production(value)


class SourceStackEffectReturnABC(SourceEventReturnABC, SourceNativeExpressionABC):
    """A source operation completed by an original consuming native transfer."""

    @property
    @abstractmethod
    def native_effect_type(self) -> type[NativeStackEffectABC]:
        raise NotImplementedError

    @property
    def receipt(self) -> NativeReturn:
        receipt = self.execution.module.native_compilation.return_after_effect(
            SourceByteSpan.require_node(self.operation.node), self.native_effect_type
        )
        receipt.frame.resolve(self)
        return receipt

    @property
    def native_effect(self) -> NativeStackEffectABC:
        return self.receipt.effect_for(
            SourceByteSpan.require_node(self.operation.node), self.native_effect_type
        )

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.receipt.require_value(value)
        _ = self.native_lookup_prefix

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_complete_source_cut(prefix)
        self.native_completion_offset(prefix)
        return self.receipt


@dataclass(frozen=True, eq=False)
class SourceDiscardReturn(SourceStackEffectReturnABC):
    """Original discarded value, proved source release and native continuation."""

    execution: SourceExecutionABC
    result: CompactEvaluatedResult
    event = AliasProperty[CompactEvaluatedResult]("result")
    native_effect_type: ClassVar[type[NativeStackEffectABC]] = NativeDiscardValue

    @property
    def source_read(self) -> CompactFlowValue:
        self.operation
        use = self.result.destination.use.require_discarded_value(self.result.value_use)
        return CompactFlowValue(self.native_frame_context, use)

    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        self.require_event_available_in(prefix, self.result)
        self.source_read
        self.execution.require_discard(cast(ast.Expr, self.operation.node))
        discard = self.native_effect
        self.require_join(discard.inputs[0])
        return discard.instruction_offset


@dataclass(frozen=True, eq=False)
class SourceFunctionReturn(SourceEventReturnABC, SourceNativeExpressionABC):
    """One returned source value joined to its function activation's native exit."""

    execution: SourceFunctionExecution
    returned: CompactEvaluatedResult
    event = AliasProperty[CompactEvaluatedResult]("returned")

    @property
    def source_read(self) -> CompactFlowValue:
        self.operation
        if self.returned.value_use is None:
            raise ValueError("Bare function return value remains unproved")
        return CompactFlowValue(self.native_frame_context, self.returned.value_use)

    @property
    def receipt(self) -> NativeReturn:
        receipt = self.execution.entry.activation.entry_continuation
        receipt.frame.resolve(self)
        return receipt

    @property
    def native_bindings(self) -> tuple[NativeBindingTransfer, ...]:
        return self.receipt.bindings

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.receipt.require_value(value)
        _ = self.native_lookup_prefix

    def _fast_local_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        return self.execution.resolve_fast_local_store(binding)

    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        self.require_event_available_in(prefix, self.returned)
        self.require_join(self.receipt.value)
        return self.receipt.instruction_offset

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        self.require_complete_source_cut(prefix)
        self.native_completion_offset(prefix)
        return self.receipt

    def result(self, prefix: AdmittedExecutionPrefixABC) -> CapturedReferenceResolution:
        self.return_continuation(prefix)
        return self.require_join(self.receipt.value)


class SourceAssignmentValueABC(SourceNativeExpressionABC):
    """An assignment's RHS belongs to its original evaluated source read."""

    binding: CompactEvaluatedAssignment

    @property
    def source_read(self) -> CompactFlowValue:
        return CompactFlowValue(self.native_frame_context, self.binding.value_use)

    @property
    def source_value(self) -> CapturedReferenceResolution:
        return self.execution.kernel.assignment_value(
            self.native_frame_context, self.binding
        )


@dataclass(frozen=True, eq=False)
class SourceItemStoreReturn(SourceStackEffectReturnABC, SourceAssignmentValueABC):
    """Original item operands joined under the source-owned storage contract."""

    execution: SourceExecutionABC
    binding: CompactEvaluatedAssignment
    event = AliasProperty[CompactEvaluatedAssignment]("binding")
    native_effect_type: ClassVar[type[NativeStackEffectABC]] = NativeItemStoreValue

    def native_completion_offset(self, prefix: AdmittedExecutionPrefixABC) -> int:
        self.require_event_available_in(prefix, self.binding)
        self.execution.require_item_write(cast(ast.Subscript, self.operation.node))
        target = cast(CompactItemTarget, self.binding.target)
        effect = self.native_effect
        value, receiver, key = effect.inputs
        self.require_join(value)
        for use, operand in (
            (target.receiver_use, receiver),
            (target.index_use, key),
        ):
            SourceNativeOperandJoin(
                self, CompactFlowValue(self.native_frame_context, use)
            ).require_join(operand)
        return effect.instruction_offset


@dataclass(frozen=True, eq=False)
class SourceAssignmentStore(SourceBindingReturnABC, SourceAssignmentValueABC):
    """One original assignment joined to its conditional native value transfer.

    The requested completed cut supplies source execution. This receipt alone
    does not establish subsequent namespace contents or class construction.
    """

    binding: CompactEvaluatedAssignment

    @property
    def production(self) -> NativeValueStore:
        name = self.binding.target.bound_name
        if name is None:
            raise ValueError("Assignment installation requires a native name binding")
        production = self.execution.source.event_operation(self.binding.value_use)
        receipt = self.execution.module.native_compilation.value_store_for(
            SourceByteSpan.require_node(production.node),
            SourceByteSpan.require_node(self.operation.node),
            name,
        )
        receipt.frame.resolve(self)
        return receipt

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.production.require_value(value)
        _ = self.native_lookup_prefix

    @property
    def native_bindings(self) -> tuple[NativeBindingTransfer, ...]:
        return self.execution.fast_local_bindings

    def require_native_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeBindingTransfer:
        return self.require_installation(prefix).binding

    def return_continuation(self, prefix: AdmittedExecutionPrefixABC) -> NativeReturn:
        """Join completed source execution to conditional later native transfers.

        A returned operand and recorded writes are not proof of their lookup or
        transfer effects, nor of native construction over the resulting namespace.
        """
        receipt = self.require_installation(prefix)
        self.require_complete_source_cut(prefix)
        return receipt.require_return()

    def require_installation(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> NativeValueStore:
        self.require_event_available_in(prefix, self.binding)
        source = self.source_value
        source.require_closed()
        receipt = self.production
        self.require_join(receipt.value)
        return receipt


@dataclass(frozen=True)
class SourceExecutionKernel(
    CapturedReferenceKernel, CompactDefinitionResolverABC[CapturedReferenceResolution]
):
    """Resolve source-created identity through the admitted creation owner."""

    effects: SourceExecutionABC

    def require_interval_frame(self, interval: SingleFlowPrefix) -> None:
        context = interval.context
        if (
            self.effects.source.compact.flow_contexts_by_identity.get(id(context))
            is not context
        ):
            raise ValueError("Interval capture requires its original source context")
        # Authenticate the frame without recursively requesting the effects proof
        # currently being closed. Scope declarations own frame construction.
        prefix = context.flow.owner.resolve_frame(
            self.effects, context, interval.position
        )
        if prefix.endpoint.frame is not interval.frame:
            raise ValueError("Interval capture requires its original activation frame")

    def _subscription_result_value_resolution(
        self,
        value: SubscriptionResultValue,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, _ = context
        operation = self.effects.source.value_operation(read)
        if not isinstance(read.use, CompactValueUse) or read.use.value is not value:
            raise ValueError("Subscription value does not belong to its original read")
        if not value.invocation.position.dominates(read.use.position):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        try:
            authority = self.effects.subscription_authority(
                read.context, value.invocation
            )
            if authority.node is not operation.node:
                raise ValueError(
                    "Subscription result belongs to a different source invocation"
                )
            return authority.result()
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, cause=error
            )

    def _tuple_value_resolution(
        self,
        value: CompactTupleValue,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, pending = context
        self.effects.source_operation(read.context, read.use)
        if not isinstance(read.use, CompactValueUse) or read.use.value is not value:
            raise ValueError("Tuple value does not belong to its original source read")
        result = SourceTupleCapture(self.effects, read, pending)
        try:
            result.require_closed()
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, cause=error
            )
        return result

    def _compiler_operand_value_resolution(
        self, value: CompilerOperandValue, context: ValueQuery
    ) -> CapturedReferenceResolution:
        read, _ = context
        self.effects.source_operation(read.context, read.use)
        if (
            not isinstance(read.use, CompactCompilerOperandUse)
            or read.use.value is not value
        ):
            raise ValueError("Compiler operand does not belong to its original read")
        try:
            return SourceCompilerOperandRead(self.effects, read).result()
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_BINDING, cause=error
            )

    def _compiler_stored_value_resolution(
        self, value: CompilerStoredValue, context: ValueQuery
    ) -> CapturedReferenceResolution:
        read, _ = context
        self.effects.source_operation(read.context, read.use)
        if not isinstance(read.use, CompactValueUse) or read.use.value is not value:
            raise ValueError(
                "Compiler value does not belong to its original source read"
            )
        result = SourceCompilerStoredCapture(self.effects, read)
        try:
            result.require_closed()
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_BINDING, cause=error
            )
        return result

    def _forwarded_result_value_resolution(
        self,
        value: ForwardedResultValue,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, pending = context
        self.effects.source_operation(read.context, read.use)
        if not isinstance(read.use, CompactValueUse) or read.use.value is not value:
            raise ValueError("Forwarded value does not belong to its original read")
        result = value.result
        self.effects.source_operation(read.context, result)
        if (
            result.value_use is None
            or not result.value_use.position.dominates(result.position)
            or not result.position.dominates(read.use.position)
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        self.effects.source_operation(read.context, result.value_use)
        prefix = self._admitted_prefix(read.context, read.use.position)
        if isinstance(prefix, OpenCapturedReference):
            return prefix
        return self._read_use(result.value_use, read.context, pending)

    def _empty_dictionary_resolution(
        self,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        read, _ = context
        try:
            result = self.effects.empty_dictionary_creation(read)
            result.require_closed()
            return result
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS,
                cause=error,
            )

    def _capture_source_read(
        self,
        context: ValueQuery,
        capture_type: type[SourceReadCaptureABC],
    ) -> CapturedReferenceResolution:
        read, _ = context
        result = capture_type(self.effects, read)
        try:
            result.require_closed()
        except (ValueError, TypeError, SyntaxError) as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_ACCESS,
                cause=error,
            )
        return result

    def _unproved_value_resolution(
        self,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        return self._capture_source_read(context, SourceLiteralCapture)

    def _function_expression_resolution(
        self,
        context: ValueQuery,
    ) -> CapturedReferenceResolution:
        return self._capture_source_read(context, SourceFunctionExpressionCapture)

    @cached_property
    def _value_resolutions(
        self,
    ) -> dict[
        tuple[SourceFlowOperation, frozenset[CompactBindingVisit[CompactFlowContext]]],
        CapturedReferenceResolution,
    ]:
        """Completed queries share canonical source/activation and cycle context."""
        return {}

    def _read_use(
        self,
        use: CompactPositionedReference,
        context: CompactFlowContext,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        operation = self.effects.source_operation(context, use)
        pending = frozenset(
            visit for visit in pending if visit.required_at_read(context, use.position)
        )
        key = (operation, pending)
        if key in self._value_resolutions:
            return self._value_resolutions[key]
        result = super()._read_use(use, context, pending)
        # An open result may reflect an in-progress recursive effect admission.
        # Never publish a partial or failed proof as a reusable completed query.
        if not isinstance(result, OpenCapturedReference):
            self._value_resolutions[key] = result
        return result

    def assignment_value(
        self,
        context: CompactFlowContext,
        binding: CompactEvaluatedAssignment,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CapturedReferenceResolution:
        """Read a stored RHS at its original capture, never at the later write cut."""
        self.effects.source_operation(context, binding)
        self.effects.source_operation(context, binding.result)
        self.effects.source_operation(context, binding.value_use)
        return self._read_use(binding.value_use, context, pending_bindings)

    def _evaluated_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactEvaluatedAssignment,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        return self.assignment_value(context, binding, pending_bindings)

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        self.effects.source_operation(context, binding)
        return binding.target.owner.resolve_definition(
            self, binding.target.bound_name, binding
        )

    def _initial_parameter_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: InitialCompactParameterBinding,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CapturedReferenceResolution:
        del pending_bindings
        prefix = self._admitted_prefix(context, use_position)
        if isinstance(prefix, OpenCapturedReference):
            return prefix
        endpoint = prefix.endpoint
        if (
            endpoint.context is not context
            or reference.attribute_path
            or binding.parameter.name != reference.root_name
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        value = prefix.entry_contents(self, endpoint.frame.locals).member(
            reference.root_name
        )
        return value or OpenCapturedReference(
            CapturedReferenceViolation.UNPROVED_BINDING
        )

    def _selected_class_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> CapturedReferenceResolution:
        node = cast(ast.ClassDef, self.effects.source.event_operation(binding).node)
        return self.effects.class_entry(node).result()

    def _selected_function_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> CapturedReferenceResolution:
        return SourceCreatedFunctionCapture.from_binding(self.effects, binding).result()


@dataclass(frozen=True)
class SourceCompletionResolver(
    CompactMutationResolverABC[CompactFlowContext, SourceCompletedReturnABC],
    CompactBindingValueResolverABC[SourceInstalledReturnABC],
    CompactDefinitionResolverABC[SourceInstalledReturnABC],
    ImportOriginResolverABC[CompactMutation, SourceInstalledReturnABC],
    CompactResultCompletionResolverABC[SourceCompletedReturnABC],
):
    """Select completion through original binding and disposition declarations."""

    execution: SourceExecutionABC

    def completed_body(
        self, context: CompactFlowContext
    ) -> SourceCompletedReturnABC | None:
        candidates: list[SourceCompletedReturnABC] = [
            self.resolve(binding) for binding in context.flow.mutations
        ]
        consumed_results = {
            id(binding.result)
            for binding in context.flow.mutations
            if isinstance(binding, CompactEvaluatedAssignment)
        }
        for result in context.flow.evaluated_results:
            if id(result) in consumed_results:
                continue
            candidates.append(result.destination.use.resolve_completion(self, result))
        selected = None
        for candidate in candidates:
            operation = candidate.operation
            position = operation.position
            if (
                candidate.native_frame_context is not context
                or position.branch_path
                or position.evaluation_path
            ):
                raise ValueError(
                    "Native completion requires one original ordered frame"
                )
            if selected is None or selected.operation.position.dominates(position):
                selected = candidate
            elif operation is not selected.operation and not position.dominates(
                selected.operation.position
            ):
                raise ValueError("Source completion boundaries have no unique order")
        return selected

    def _returned_result_completion(
        self, result: CompactEvaluatedResult
    ) -> SourceCompletedReturnABC:
        self.execution.source.event_operation(result)
        if not isinstance(self.execution, SourceFunctionExecution):
            raise ValueError("Returned completion requires a function activation")
        return SourceFunctionReturn(self.execution, result)

    def _discarded_result_completion(
        self, result: CompactEvaluatedResult
    ) -> SourceCompletedReturnABC:
        operation = self.execution.source.event_operation(result)
        dispositions = result.destination.use.expression_operations(
            result, self.execution.source.operations_by_node[operation.node]
        )
        if dispositions[0].event is not result:
            raise ValueError("Discard completion requires its original disposition")
        return SourceDiscardReturn(self.execution, result)

    def _deleted_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> SourceInstalledReturnABC:
        return SourceDeletionReturn(self.execution, binding)

    def resolve(self, binding: CompactMutation) -> SourceCompletedReturnABC:
        operation = self.execution.source.event_operation(binding)
        context = self.execution.context_for_owner(operation.owner)
        return binding.resolve(self, context)

    def _binding_mutation_resolution(
        self, context: CompactFlowContext, binding: CompactMutation, name: str
    ) -> SourceInstalledReturnABC:
        reference = binding.reference
        if reference is None:
            raise ValueError("Native continuation requires an original lexical binding")
        return binding.kind.binding_operation.resolve_source(
            self, context, reference, binding, frozenset()
        )

    def _item_mutation_resolution(
        self, context: CompactFlowContext, binding: CompactMutation[CompactItemTarget]
    ) -> SourceCompletedReturnABC:
        if not isinstance(binding, CompactEvaluatedAssignment):
            raise ValueError("Item completion requires an evaluated source assignment")
        return SourceItemStoreReturn(self.execution, binding)

    def _receiver_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> SourceCompletedReturnABC:
        raise ValueError("Native receiver completion remains unproved")

    def _evaluated_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactEvaluatedAssignment,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> SourceInstalledReturnABC:
        return SourceAssignmentStore(self.execution, binding)

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> SourceInstalledReturnABC:
        return binding.kind.binding_operation.resolve_definition(
            self, reference.root_name, binding
        )

    def _selected_function_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> SourceInstalledReturnABC:
        return SourceCreatedFunctionCapture.from_binding(
            self.execution, binding
        ).creation_results[-1]

    def _selected_class_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> SourceInstalledReturnABC:
        node = cast(ast.ClassDef, self.execution.source.event_operation(binding).node)
        return self.execution.class_entry(node).result()

    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> SourceInstalledReturnABC:
        origin = cast(CompactImportTarget, binding.target).origin
        return origin.resolve(self, binding)

    def _module_import_resolution(
        self, origin: ImportedNameOrigin, binding: CompactMutation
    ) -> SourceInstalledReturnABC:
        return SourceModuleImportStore(self.execution, binding)

    def _member_import_resolution(
        self, origin: ImportedNameOrigin, binding: CompactMutation
    ) -> SourceInstalledReturnABC:
        return SourceMemberImportStore(self.execution, binding)

    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> SourceInstalledReturnABC:
        raise ValueError(
            f"Native store continuation remains unproved: {violation.value}"
        )


@dataclass(frozen=True, eq=False)
class SourceClassBodyEntryABC(
    SourceDefinitionEntry[ast.ClassDef],
    QualifiedDeclaration,
    NamespaceCreationEvidenceABC,
    NativeClassCaptureResolverABC[ExactNativeClassCapture],
    NativeClassPrologueResolverABC[dict[NativeScalar, CapturedReferenceResolution]],
    RecordedNamespace,
    SourceNativeNamespaceABC,
):
    """Shared source-bound class-body and prepared-storage evidence.

    The native builder and body creation are checked at their original captures.
    Deferred method bodies do not enter this activation. Unknown generated frame
    origins, custom preparation and prologue writes outside locals stay open.
    """

    native_builder: ClassVar[NativeDeclaration] = NativeDeclaration(
        builtins.__build_class__
    )

    native_lookup_prefix = AliasProperty[AdmittedExecutionPrefixABC]("parent_prefix")
    globals = AliasProperty[NamespaceEvidenceABC | OpenCapturedReference](
        "parent_prefix.endpoint.frame.globals"
    )

    native_bindings = AliasProperty[tuple[NativeBindingTransfer, ...]](
        "capture.prologue.bindings"
    )

    @cached_property
    def annotation_namespace(self) -> SourceAnnotationNamespace:
        values = tuple(
            value
            for value in self.capture.prologue.values
            if isinstance(value, NativeAnnotationNamespaceValue)
        )
        if len(values) != 1:
            raise ValueError("Class annotation namespace has no unique original setup")
        return SourceAnnotationNamespace(self, values[0])

    @property
    @abstractmethod
    def construction_admission(self) -> None:
        """Require the selected native construction over its actual completed body."""
        raise NotImplementedError

    @property
    def completed(self) -> None:
        """Validate the original source boundary before reusing construction proof."""
        _ = self.frame
        _ = self.final_evaluation
        _ = self.construction_admission

    @property
    def native_tail(self) -> PreparedNamespaceContinuationABC:
        _ = self.final_evaluation
        completion = SourceCompletionResolver(self.execution).completed_body(
            self.context
        )
        tail = (
            PreparedNamespaceTail(self, completion)
            if completion is not None
            else EntryOnlyNamespaceContinuation(self)
        )
        _ = tail.completed
        return tail

    @property
    def native_class_entry(self) -> SourceClassBodyEntryABC:
        return self

    def _native_initial_local(
        self, name: NativeScalar
    ) -> CapturedReferenceResolution | None:
        return None

    @property
    def native_global_namespaces(
        self,
    ) -> tuple[NamespaceEvidenceABC | OpenCapturedReference, ...]:
        return self.globals, self.builtins

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.capture.prologue.require_value(value)
        _ = self.parent_prefix

    @property
    def final_evaluation(self) -> SourceFlowEvaluation:
        """Join the original final statement, including event-free statements.

        Source completion does not establish native tail execution or construction.
        The retained syntax order prevents a mutated AST from dropping a trailing
        pass or selecting an earlier store as the original completion boundary.
        """
        self.require_original_operation()
        source = self.execution.source
        syntax = module_syntax_index(source.module.module)
        original_body = tuple(
            node
            for node in syntax.children_by_node.get(self.node, ())
            if isinstance(node, ast.stmt)
        )
        if (
            not original_body
            or len(original_body) != len(self.node.body)
            or any(
                original is not current
                for original, current in zip(original_body, self.node.body, strict=True)
            )
        ):
            raise ValueError("Class completion requires its original body statements")
        evaluations = tuple(
            evaluation
            for evaluation in source.evaluation_bounds_by_node.get(
                original_body[-1], ()
            )
            if evaluation.owner is self.context.flow.owner
        )
        if len(evaluations) != 1:
            raise ValueError("Final statement has no unique original evaluation")
        evaluation = evaluations[0]
        self.execution.require_terminal_evaluation(evaluation)
        return evaluation

    @abstractmethod
    def require_native_preparation(self) -> None:
        """Discharge the selected native fresh empty exact-dict preparation law."""
        raise NotImplementedError

    @classmethod
    def for_definition(
        cls, execution: SourceExecutionABC, node: ast.ClassDef
    ) -> SourceClassBodyEntryABC:
        """Select actual header evidence before publishing one canonical entry."""
        execution.definition_operation(node)
        operand = AstClassProjection.explicit_metaclass(node)
        if operand is None:
            return SourceClassEntry(execution, node)
        family = NativeSourceClassEntryABC.select_from_capture(
            execution.capture_value(operand)
        )
        return family(execution, node)

    @abstractmethod
    def _created_result(self) -> SourceDefinitionResultABC:
        """Retain the result protocol selected by this entry's native construction."""
        raise NotImplementedError

    @cached_property
    def created_result(self) -> SourceDefinitionResultABC:
        return self._created_result()

    @cached_property
    def creation_results(self) -> tuple[SourceDefinitionResultABC, ...]:
        return (
            self.created_result,
            *(
                SourceClassDecoratorApplication(self, index)
                for index in range(len(self.definition.target.decorator_uses))
            ),
        )

    def result(self) -> SourceDefinitionResultABC:
        """Expose the construction-owned result only after its final source binding."""
        self.require_installed_result()
        return self.creation_results[-1]

    def require_installed_result(self) -> None:
        """Close transformation and final storage separately from raw construction."""
        _ = self.completed
        if self.definition.target.decorator_uses:
            self.creation_results[-1].require_closed()
        self.execution.require_binding_write(self.node)

    def require_preparation(self) -> None:
        """Authenticate original builder/body captures before native preparation."""
        _ = self.builtins
        self.require_native_preparation()

    def completion_member(self, name: str) -> CapturedReferenceResolution:
        """Resolve actual prepared storage at the original completed-body cut."""
        value = self.execution.kernel._namespace_resolution(
            self, name, self.completion_prefix, frozenset()
        )
        if value is None:
            raise ValueError("Complete namespace inventory has no member value")
        return value

    def require_native_creator(self, execution: ExactNativeFunctionExecution) -> None:
        self.require_admitted(self.initial)
        if execution is not self.capture.body:
            raise ValueError("Native creator belongs to a different class body")

    @cached_property
    def completion_prefix(self) -> AdmittedExecutionPrefixABC:
        """Retain completed descendants through the canonical sequential prefix."""
        return self.execution.required_prefix(self.context, None)

    @property
    def qualified_name(self) -> str:
        return f"{self.execution.module.module_name}.{self.context.flow.owner.qualname}"

    @property
    def initial(self) -> InitialNativeIsland:
        return self.execution.initial

    @cached_property
    def capture(self) -> ExactNativeClassCapture:
        return self.context.flow.owner.capture.resolve(self)

    def _open_class_capture_resolution(
        self, capture: OpenNativeClassCapture
    ) -> ExactNativeClassCapture:
        raise ValueError("Native class capture remains unproved")

    def capture_operation(self, native_site: NativeCaptureSite) -> SourceFlowOperation:
        sites = tuple(
            site
            for site in self.execution.source.operations
            if site.node is self.node
            and site.owner is self.operation.owner
            and isinstance(site.event, CompactNativeCapture)
            and site.event.site is native_site
        )
        if len(sites) != 1:
            raise ValueError("Native class capture has no unique source operation")
        native_site.frame.resolve(self)
        return sites[0]

    def _generated_class_frame_origin_resolution(
        self, origin: GeneratedClassNativeFrameOrigin
    ) -> CompactFlowContext:
        capture = self.capture
        if (
            capture.builder.frame is not origin
            or capture.creation.frame is not origin
            or origin.execution.source_span != capture.source_span
            or origin.activation.frame is not origin.execution.creation.frame
            or origin.activation.instruction_offset
            <= origin.execution.creation.instruction_offset
        ):
            raise ValueError("Generated class frame differs from its source entry")
        origin.activation.frame.resolve(self)
        return self.parent_context

    def _class_closure_native_value_resolution(
        self, value: NativeClassClosureValue
    ) -> CapturedReferenceResolution:
        origin = self.capture.creation.frame
        origin.resolve(self)
        self.require_preparation()
        declaration = origin.class_closure_declaration(value)
        return NativeTypePremise(declaration.declaration)

    @cached_property
    def builder_value(self) -> CapturedReferenceResolution:
        builder = self.capture_operation(self.capture.builder)
        builder_prefix = self.execution.required_prefix(
            self.parent_context, builder.position
        )
        value = self.execution.kernel._slot(
            builder_prefix.endpoint.frame.builtins,
            self.native_builder.declaration.__name__,
            self.parent_context,
            builder.position,
            frozenset(),
        )
        if value is None:
            raise ValueError("Native builder slot is absent")
        value.require_native_identity(self.native_builder)
        return value

    @cached_property
    def builtins(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        _ = self.builder_value
        creation = self.capture_operation(self.capture.creation)
        prefix = self.execution.required_prefix(self.parent_context, creation.position)
        value = self.execution.kernel._namespaces_resolution(
            prefix,
            (prefix.endpoint.frame.globals,),
            self.execution.entry.bootstrap_binding_name(),
            frozenset(),
        )
        return value.as_builtin_namespace(self.initial)

    @cached_property
    def initial_entries(self) -> Mapping[NativeScalar, CapturedReferenceResolution]:
        return self.capture_initial_entries(self.capture.prologue.resolve(self))

    def _exact_class_prologue_resolution(
        self, prologue: ExactNativeClassPrologue
    ) -> dict[NativeScalar, CapturedReferenceResolution]:
        for value in prologue.values:
            self.native_value(value).require_closed()
        return dict(
            resolved
            for binding in prologue.bindings
            if (resolved := binding.resolve(self)) is not None
        )

    def _open_class_prologue_resolution(
        self, prologue: OpenNativeClassPrologue
    ) -> dict[NativeScalar, CapturedReferenceResolution]:
        raise ValueError("Native class prologue remains unproved")

    def _cell_store_resolution(self, binding: NativeBindingTransfer) -> None:
        self.capture.prologue.require_fresh_cell_store(binding)

    def _cell_creation_resolution(self, binding: NativeBindingTransfer) -> None:
        # The native MAKE_CELL receipt creates frame-owned storage, not a key
        # in the prepared class dictionary. Later stores retain that relation.
        pass

    @cached_property
    def frame(self) -> InitialNativeFrame:
        _ = self.initial_entries
        self.require_preparation()
        return InitialNativeFrame(self, self.globals, self.builtins)

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if initial is not self.initial:
            raise ValueError("Source class belongs to a foreign native admission")

    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        return self.initial_entries.get(key)

    def prefix(
        self, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC:
        return ChildExecutionPrefix(
            self.parent_prefix,
            self.definition,
            CapturedFlowPrefix(
                self.context, self.frame, position, kernel=self.execution.kernel
            ),
        )


@dataclass(frozen=True, eq=False)
class PreparedNamespaceContinuationABC(SourceNativeNamespaceABC):
    """Interpret original native work after a proved source-body boundary."""

    entry: SourceClassBodyEntryABC

    execution = AliasProperty["SourceExecutionABC"]("entry.execution")
    native_lookup_prefix = AliasProperty[AdmittedExecutionPrefixABC](
        "entry.completion_prefix"
    )
    native_global_namespaces = AliasProperty[
        tuple[NamespaceEvidenceABC | OpenCapturedReference, ...]
    ]("entry.native_global_namespaces")
    native_bindings = AliasProperty[tuple[NativeBindingTransfer, ...]]("bindings")
    native_class_entry = AliasProperty[SourceClassBodyEntryABC]("entry")

    def require_member(self, name: NativeScalar) -> CapturedReferenceResolution:
        value = self.member(name)
        if value is None:
            raise ValueError("Complete native namespace inventory has no member value")
        return value

    @property
    def names(self) -> frozenset[NativeScalar]:
        _ = self.completed
        source = NamespaceMemberInventory(
            self.execution.kernel, self.entry, self.native_lookup_prefix
        ).names
        return source | frozenset(binding.name for binding in self.native_bindings)

    def _require_native_production(self, value: NativeProducedValue) -> None:
        self.receipt.require_value(value)
        if not self.follows_source(value.instruction_offset):
            raise ValueError(
                "Native tail value precedes the completed source store or body boundary"
            )

    @property
    def completed(self) -> None:
        receipt = self.receipt
        for value in receipt.values:
            if self.follows_source(value.instruction_offset):
                value.require_completion(self)
        for binding in self.native_bindings:
            binding.resolve(self)
        receipt.value.require_completion(self)

    def _native_initial_local(
        self, name: NativeScalar
    ) -> CapturedReferenceResolution | None:
        return self.execution.kernel._namespace_resolution(
            self.entry, name, self.native_lookup_prefix, frozenset()
        )

    def member(self, name: NativeScalar) -> CapturedReferenceResolution | None:
        _ = self.completed
        return self._preceding_native_local_value(name, self.receipt.instruction_offset)

    @property
    @abstractmethod
    def receipt(self) -> NativeReturn:
        raise NotImplementedError

    @abstractmethod
    def follows_source(self, instruction_offset: int) -> bool:
        raise NotImplementedError

    @property
    def bindings(self) -> tuple[NativeBindingTransfer, ...]:
        return tuple(
            binding
            for binding in self.receipt.bindings
            if self.follows_source(binding.instruction_offset)
        )


@dataclass(frozen=True, eq=False)
class PreparedNamespaceTail(PreparedNamespaceContinuationABC):
    """Native work after the last source completion in one completed prepared namespace.

    Local transfers use completed source storage and preceding native writes.
    Original cell creation and overwritten-value lifetime remain checked.
    Construction over the resulting namespace is a separate obligation.
    """

    completion: SourceCompletedReturnABC

    def follows_source(self, instruction_offset: int) -> bool:
        return instruction_offset > self.completion.native_completion_offset(
            self.native_lookup_prefix
        )

    @property
    def receipt(self) -> NativeReturn:
        entry = self.entry
        completion = self.completion
        if (
            completion.execution is not entry.execution
            or completion.native_frame_context is not entry.context
        ):
            raise ValueError(
                "Prepared namespace tail belongs to a different source frame"
            )
        _ = entry.final_evaluation
        prefix = entry.completion_prefix
        receipt = completion.return_continuation(prefix)
        if (
            entry.execution.require_terminal_evaluation(
                completion.completion_evaluation
            )
            is not prefix
        ):
            raise ValueError(
                "Prepared namespace tail has a different completed source cut"
            )
        return receipt


class EntryOnlyNamespaceContinuation(PreparedNamespaceContinuationABC):
    """An event-free original source body joins the compiler's prologue boundary."""

    def native_value(self, value: NativeProducedValue) -> CapturedReferenceResolution:
        if self.follows_source(value.instruction_offset):
            return super().native_value(value)
        # This value was already produced in the original prologue. Resolve it in
        # that context, not by replaying its read against completed storage.
        original = self.execution.module.native_compilation.prologue_return_operand(
            self.entry.capture, self.receipt, value
        )
        return self.entry.native_value(original)

    def follows_source(self, instruction_offset: int) -> bool:
        return instruction_offset >= self.entry.capture.prologue.require_body_start()

    @property
    def receipt(self) -> NativeReturn:
        entry = self.entry
        _ = entry.final_evaluation
        prefix = entry.completion_prefix
        entry.execution.require_empty_interval(
            SingleFlowPrefix(entry.context, prefix.endpoint.frame, None)
        )
        receipt = entry.execution.module.native_compilation.return_from(
            entry.capture.body
        )
        # The shared observer must cover the entire native body, not a later suffix.
        start = entry.capture.prologue.require_body_start()
        if not any(store.instruction_offset < start for store in receipt.stores):
            raise ValueError(
                "Native continuation does not cover the original body boundary"
            )
        return receipt


class SourceClassEntry(SourceClassBodyEntryABC):
    """Ordinary native type construction over the actual prepared body."""

    def require_native_preparation(self) -> None:
        """Retain ordinary base/MRO admission before exposing the frame."""
        _ = self.mro_type

    def _created_result(self) -> SourceCreatedClassCapture:
        return SourceCreatedClassCapture(self)

    def require_absent_native_hooks(self, hooks: tuple[NativeDeclaration, ...]) -> None:
        """Inspect admitted type-construction inputs, not a live class dictionary.

        Source class capture excludes replacing builders/metaclasses and slots.
        The caller's admitted prefix separately rejects writes to created classes.
        This receipt does not expose the prepared dictionary as later type storage.
        """
        names = frozenset(hook.declaration.__name__ for hook in hooks)
        for ancestor in self.mro_type.declarations:
            actual = NamespaceMemberInventory(
                self.execution.kernel,
                ancestor,
                ancestor.completion_prefix,
            ).names
            unproved = names & actual
            if unproved:
                raise ValueError(
                    f"Source class {ancestor.qualified_name!r} has unproved native protocol hooks: "
                    + ", ".join(sorted(unproved))
                )

    def _exact_class_capture_resolution(
        self, capture: ExactNativeClassCapture
    ) -> ExactNativeClassCapture:
        if self.node.keywords:
            raise ValueError("Native class entry has unproved construction hooks")
        return capture

    @cached_property
    def mro_type(self) -> DeclarationMroType[SourceClassEntry]:
        bases = tuple(
            self.execution.capture(base).require_plain_class_base(
                self.execution.kernel,
                self.parent_context,
                self.definition.target.header_position,
            )
            for base in self.node.bases
        )
        try:
            return DeclarationMroType.from_declaration(self, bases)
        except TypeError as error:
            raise ValueError(
                "Native source class hierarchy remains unproved"
            ) from error

    @cached_property
    def plain_subclass_protocol(self) -> None:
        self.require_absent_native_hooks((NativeDeclaration(object.__init_subclass__),))

    @cached_property
    def construction_admission(self) -> None:
        """Complete raw native construction over actual final namespace values."""
        tail = self.native_tail
        names = tail.names
        for requirement in NativeCreationBackend.current().class_construction_fields(
            names
        ):
            requirement.require_value(tail.require_member(requirement.value), self)
        for name in names:
            tail.require_member(name).require_class_installation()


@dataclass(frozen=True, eq=False)
class SourceClassDecoratorApplication(
    SourceDefinitionCapture,
    OpaqueCapturedObjectOperations,
    SourceDefinitionDecoratorApplicationABC,
):
    """Class decorator topology without a transformation or identity claim."""

    creation: SourceClassBodyEntryABC
    index: int
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    @property
    def _native_application_path(
        self,
    ) -> tuple[NativeProducedValue, tuple[NativeCallValue, ...]]:
        store = self.production
        raw = self.creation.capture.definition_construction_in(store)
        return raw, store.applications_after(raw)

    def _require_native_application_predecessors(
        self,
        raw: NativeProducedValue,
        applications: tuple[NativeCallValue, ...],
    ) -> None:
        preceding = raw
        for application in applications:
            if application.require_definition_argument() is not preceding:
                raise ValueError(
                    "Native class application differs from its stored predecessor chain"
                )
            preceding = application

    @cached_property
    def application_authority(self) -> NativeDefinitionApplicationAuthorityABC:
        return self.callee.definition_application_authority(self)

    def require_closed(self) -> None:
        try:
            result = self.application_authority.result()
        except ValueError as error:
            raise ValueError("Class decorator result remains unproved") from error
        if result is not self.argument:
            raise ValueError("Class decorator returned a different definition object")

    def native_definition_application_authority(
        self, callee: CapturedNativeObject
    ) -> NativeDefinitionApplicationAuthorityABC:
        return NativeDefinitionApplicationAuthorityABC.for_application(self, callee)

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        """Delegate only through the selected transformation's base-preservation law."""
        self.require_closed()
        return self.application_authority.require_plain_class_base(
            resolver,
            context,
            position,
        )


class NativeSourceClassEntryABC(SourceClassBodyEntryABC, NativeDeclarationFamily):
    """Original explicit native metaclass with independently proved preparation.

    The complete prepared body uses the shared source namespace authority.
    Construction and its installed result remain separate native obligations.
    """

    @cached_property
    def metaclass_declaration(self) -> NativeDeclaration:
        self.require_original_operation()
        operand = AstClassProjection.require_explicit_metaclass(self.node)
        read = self.execution.source.value_reads_by_node[operand]
        if read.context is not self.parent_context or not any(
            read.use is original for original in self.definition.target.input_uses
        ):
            raise ValueError("Metaclass requires its original header input")
        return self.execution.capture_value(operand).require_native(
            self.native_declarations
        )

    def _exact_class_capture_resolution(
        self, capture: ExactNativeClassCapture
    ) -> ExactNativeClassCapture:
        if self.node.bases or len(self.node.keywords) != 1:
            raise ValueError("Native metaclass header binding remains unproved")
        _ = self.metaclass_declaration
        return capture

    def require_native_preparation(self) -> None:
        self.execution.require_native_behavior(self)
        NativeCreationBackend.current().require_fresh_class_namespace(
            NativeClassMroDeclaration(
                cast(type, self.metaclass_declaration.declaration)
            )
        )

    @cached_property
    def construction_admission(self) -> None:
        raise ValueError(
            "Native class construction over prepared inputs remains unproved"
        )


class AutoRegisterClassEntry(NativeSourceClassEntryABC):
    """Source preparation selected by the actual AutoRegisterMeta declaration."""

    native_declarations = (AutoRegisterClassAuthority.native_metaclass,)

    def require_concrete_prepared_members(self) -> None:
        """Check original prepared values; this supplies no class result identity."""
        tail = self.native_tail
        for name in tail.names:
            tail.require_member(name).require_nonabstract_member(self.completion_prefix)

    @property
    def construction_admission(self) -> None:
        self.require_concrete_prepared_members()
        return super().construction_admission

    def _created_result(self) -> SourceDefinitionResultABC:
        raise ValueError("Native registration class result remains unproved")


@dataclass(eq=False)
class SourceExecutionABC(
    NativeReferenceEnvironment,
    FlowFrameResolverABC[AdmittedExecutionPrefixABC],
    CapturedReferenceEffectsABC,
    CompactMutationResolverABC[CompactFlowContext, None],
):
    """One activation's original source flow, effects and completed proofs.

    The entry owns its namespace and prefix convention. Caches and the kernel
    belong to this execution, never to a structurally equal source activation.
    Closure queries historical cuts, never an optimistic active proof.
    """

    entry: SourceExecutionEntryABC

    module = AliasProperty[ParsedModule]("entry.source.module")
    source = AliasProperty[SourceProductFlowProjection]("entry.source")
    initial = AliasProperty[InitialNativeIsland]("entry.initial")

    native_import: ClassVar[NativeDeclaration] = NativeDeclaration(builtins.__import__)

    @property
    def fast_local_bindings(self) -> tuple[NativeBindingTransfer, ...]:
        """Module and class activations do not own CPython fast-local storage."""
        return ()

    def _entry_flow_frame(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC:
        self.entry.require_context(context)
        return self.entry.prefix(position, self.kernel)

    _function_flow_frame = _namespace_flow_frame = _entry_flow_frame

    @property
    @abstractmethod
    def initial_contents(self) -> NamespaceContentsABC:
        """Project the actual entry's initial native work without changing its facts."""
        raise NotImplementedError

    def _class_flow_frame(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC:
        try:
            parent, binding = self.source.compact.definition_sources_by_owner[
                context.flow.owner
            ]
        except KeyError as error:
            raise ValueError("Source activation remains unproved") from error
        operation = self.source.source_operation(parent, binding)
        entry = self.class_entry(operation.node)
        if entry.context is not context or entry.definition is not binding:
            raise ValueError("Source activation remains unproved")
        return entry.prefix(position)

    def entry_contents(
        self,
        kernel: CapturedReferenceKernel,
        namespace: NamespaceEvidenceABC,
        prefix: AdmittedExecutionPrefixABC,
    ) -> NamespaceContentsABC:
        if namespace is self.entry:
            if kernel is not self.kernel:
                raise ValueError("Entry contents require their original kernel")
            self.entry.require_available(kernel, prefix)
            return self.initial_contents
        return super().entry_contents(kernel, namespace, prefix)

    _closed_storage_operations: set[SourceFlowOperation] = field(
        default_factory=set, init=False
    )

    require_operation_completion = AliasProperty[
        Callable[[SourceActivationAuthorityABC], None]
    ]("entry.require_operation_completion")

    require_native_behavior = AliasProperty[
        Callable[[SourceActivationAuthorityABC], None]
    ]("entry.require_native_behavior")

    def require_empty_interval(self, interval: SingleFlowPrefix) -> None:
        """Require no original source operations or effects within this actual interval."""
        if any(
            operation.owner is interval.context.flow.owner
            and interval.contains(operation.event)
            for operation in self.source.operations
        ) or tuple(self.effects.occurrences(self.source, interval)):
            raise ValueError("Later original source operations or effects remain")

    def item_installation_value(
        self,
        occurrence: ContextualMutation,
    ) -> CapturedReferenceResolution:
        """Recover a stored RHS only after its actual native setter is admitted."""
        context = occurrence.source.context
        mutation = occurrence.mutation
        operation = self.source_operation(context, mutation)
        if not isinstance(mutation, CompactEvaluatedAssignment) or not isinstance(
            mutation.target, CompactItemTarget
        ):
            raise TypeError(
                "Item installation requires its original evaluated item store"
            )
        occurrence.source.require_definite(mutation)
        prefix = self.required_prefix(context, mutation.position)
        if occurrence.source.frame is not prefix.endpoint.frame:
            raise ValueError(
                "Item installation belongs to a different activation frame"
            )
        self._require_storage_operation(operation)
        return self.kernel.assignment_value(context, mutation)

    def require_import_operation(self, operation: SourceFlowOperation) -> None:
        mutation = operation.event
        if not isinstance(mutation, CompactMutation) or not isinstance(
            mutation.target, CompactImportTarget
        ):
            raise ValueError("Import proof requires an actual import binding operation")
        context = self.context_for_owner(operation.owner)
        if self.source_operation(context, mutation) is not operation:
            raise ValueError("Import proof requires the original source operation")
        prefix = self.required_prefix(context, operation.position)
        importer = self.kernel._slot(
            prefix.endpoint.frame.builtins,
            self.native_import.declaration.__name__,
            context,
            operation.position,
            frozenset(),
        )
        if importer is None:
            raise ValueError("Native import slot is absent")
        importer.require_native_identity(self.native_import)
        mutation.target.origin.resolve(
            self.kernel, (context, mutation, frozenset())
        ).require_closed()
        self._require_binding_operation(operation)

    def _require_binding_operation(self, operation: SourceFlowOperation) -> None:
        mutation = operation.event
        if not isinstance(mutation, CompactMutation) or not isinstance(
            mutation.target, CompactLexicalBindingTargetABC
        ):
            raise ValueError("Binding proof requires a lexical mutation operation")
        self._require_storage_operation(operation)

    def _require_storage_operation(self, operation: SourceFlowOperation) -> None:
        """Authenticate once and retain only a completed target-owned storage proof."""
        mutation = operation.event
        if not isinstance(mutation, CompactMutation):
            raise ValueError("Storage proof requires an actual mutation")
        context = self.context_for_owner(operation.owner)
        if self.source_operation(context, mutation) is not operation:
            raise ValueError("Storage proof requires the original source operation")
        if operation in self._closed_storage_operations:
            return
        mutation.resolve(self, context)
        self._closed_storage_operations.add(operation)

    def _binding_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation,
        name: str,
    ) -> None:
        prefix = self.required_prefix(context, mutation.position)
        namespace = prefix.endpoint.frame.binding_namespace(context, name)
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
        previous = namespace.require_slot_release(
            self.kernel, name, context, mutation.position
        )
        mutation.kind.binding_operation.require_previous_binding(previous is not None)

    def _attribute_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> None:
        mutation.kind.binding_operation.require_plain_store()
        receiver = self.kernel._read_use(
            mutation.target.receiver_use, context, frozenset()
        )
        receiver.require_attribute_write(
            self.kernel, mutation.target.attribute_name, context, mutation.position
        )

    def _item_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation[CompactItemTarget],
    ) -> None:
        if not isinstance(mutation, CompactEvaluatedAssignment):
            raise ValueError(
                "Source item store requires an actual evaluated item assignment"
            )
        self.kernel.assignment_value(context, mutation).require_closed()
        receiver = self.kernel._read_use(
            mutation.target.receiver_use, context, frozenset()
        )
        key = self.kernel._read_use(
            mutation.target.index_use, context, frozenset()
        ).require_native_scalar()
        receiver.require_item_write(self.kernel, key, context, mutation.position)

    def _receiver_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> None:
        raise ValueError("Source receiver storage protocol remains unproved")

    _class_entries: dict[ast.ClassDef, SourceClassBodyEntryABC] = field(
        default_factory=dict, init=False
    )
    _pending: set[tuple[int, CompactFlowPosition | None]] = field(
        default_factory=set, init=False
    )
    _admissions: dict[
        tuple[int, CompactFlowPosition | None],
        AdmittedExecutionPrefixABC | OpenCapturedReference,
    ] = field(default_factory=dict, init=False)

    _closed_intervals: set[SingleFlowPrefix] = field(default_factory=set, init=False)

    def require_discard(self, node: ast.Expr) -> None:
        operations = tuple(
            operation
            for operation in self.source.operations_by_node.get(node, ())
            if isinstance(operation.event, CompactEvaluatedResult)
        )
        if len(operations) != 1:
            raise ValueError("Discard has no unique original evaluated result")
        operation = operations[0]
        context = self.context_for_owner(operation.owner)
        result = cast(CompactEvaluatedResult, operation.event)
        self.source_operation(context, result)
        value_use = result.destination.use.require_discarded_value(result.value_use)
        prefix = self.required_prefix(context, result.position)
        try:
            LiteralExpressionEffects(node.value).require_closed()
        except (ValueError, TypeError, SyntaxError):
            value = self.kernel._read_use(value_use, context, frozenset())
            value.require_closed()
            value.require_release_in(prefix.endpoint.frame)

    def _preceding_class_entries(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> tuple[SourceClassBodyEntryABC, ...]:
        """Derive the mandatory prior activations in their actual producer order."""
        entries = []
        for event in context.flow.mutations:
            if not (
                isinstance(event.target, CompactDefinitionTarget)
                and isinstance(event.target.owner, CompactClassDeclaration)
                and (position is None or event.position.dominates(position))
            ):
                continue
            if event.position.branch_path or event.position.evaluation_path:
                raise ValueError(
                    "Conditional or repeated source class activation remains unproved"
                )
            entries.append(self.class_entry(self.source_operation(context, event).node))
        return tuple(entries)

    def require_call(self, node: ast.Call) -> None:
        context, call = self.source_call(node)
        self.required_prefix(context, call.position)
        self.call_authority(context, call).require_closed()

    def call_result(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> CapturedReferenceResolution:
        self.call_site(context, call)
        try:
            self.required_prefix(context, call.position)
            authority = self.call_authority(context, call)
            authority.require_closed()
            return authority.captured_result
        except ValueError as error:
            return OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS, cause=error
            )

    def source_call(
        self, node: ast.Call
    ) -> tuple[CompactFlowContext, CompactFunctionCall]:
        operations = tuple(
            operation
            for operation in self.source.operations_by_node.get(node, ())
            if isinstance(operation.event, CompactFunctionCall)
        )
        if len(operations) != 1:
            raise ValueError("Call operand has no unique original invocation")
        operation = operations[0]
        context = self.context_for_owner(operation.owner)
        call = cast(CompactFunctionCall, operation.event)
        if self.call_site(context, call) is not node:
            raise ValueError("Call operand is not the original source node")
        return context, call

    def call_site(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> ast.Call:
        operation = self.source_operation(context, call)
        if not isinstance(operation.node, ast.Call) or not any(
            candidate is call for candidate in context.flow.calls
        ):
            raise ValueError("Call has no original canonical invocation")
        return operation.node

    @cached_property
    def effects(self) -> SourceExecutionEffectEvidence:
        return SourceExecutionEffectEvidence.from_source(self.module.module)

    @cached_property
    def kernel(self) -> SourceExecutionKernel:
        return SourceExecutionKernel(self.initial, self)

    @cached_property
    def _empty_dictionary_creations(
        self,
    ) -> dict[SourceFlowOperation, SourceEmptyDictionaryCapture]:
        """Creation owners share this execution lifetime; no slot contents are stored."""
        return {}

    def empty_dictionary_creation(
        self,
        read: CompactFlowValue,
    ) -> SourceEmptyDictionaryCapture:
        operation = self.source.value_operation(read)
        if not isinstance(
            cast(CompactValueUse, read.use).value, EmptyDictionaryExpression
        ):
            raise ValueError(
                "Empty dictionary requires its unique original source value"
            )
        if operation not in self._empty_dictionary_creations:
            self._empty_dictionary_creations[operation] = SourceEmptyDictionaryCapture(
                self, operation
            )
        return self._empty_dictionary_creations[operation]

    def class_entry(self, node: ast.ClassDef) -> SourceClassBodyEntryABC:

        if node not in self._class_entries:
            self._class_entries[node] = SourceClassBodyEntryABC.for_definition(
                self, node
            )
        return self._class_entries[node]

    def require_terminal_evaluation(
        self, evaluation: SourceFlowEvaluation
    ) -> AdmittedExecutionPrefixABC:
        """Require original completed work with no subsequent source operations or effects."""
        if not any(
            original is evaluation
            for original in self.source.evaluation_bounds_by_node.get(
                evaluation.node, ()
            )
        ):
            raise ValueError("Completion requires its original source evaluation")
        context = self.context_for_owner(evaluation.owner)
        prefix = self.required_prefix(context, None)
        remainder = SingleFlowPrefix(
            context, prefix.endpoint.frame, None, evaluation.exit
        )
        self.require_empty_interval(remainder)
        return prefix

    def required_prefix(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC:
        result = self.admit(context, position)
        if isinstance(result, OpenCapturedReference):
            result.require_closed()
        return result

    def admit(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC | OpenCapturedReference:
        if (
            self.source.compact.flow_contexts_by_identity.get(id(context))
            is not context
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        key = (id(context), position)
        if key in self._admissions:
            return self._admissions[key]
        if key in self._pending:
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        self._pending.add(key)
        try:
            prefix = self._prefix(context, position)
            for interval in prefix.intervals:
                interval.capture_kernel(self.kernel).effects.require_interval_effects(
                    interval
                )
            prefix.require_admitted(self.initial)
            result = prefix
        except ValueError as error:
            result = OpenCapturedReference(
                CapturedReferenceViolation.UNPROVED_EFFECTS,
                cause=error,
            )
        finally:
            self._pending.remove(key)
        self._admissions[key] = result
        return result

    def _prefix(
        self, context: CompactFlowContext, position: CompactFlowPosition | None
    ) -> AdmittedExecutionPrefixABC:
        completed = self._preceding_class_entries(context, position)
        for entry in completed:
            _ = entry.completed
        prefix = context.flow.owner.resolve_frame(self, context, position)
        if completed:
            entry = completed[-1]
            last = entry.definition
            previous = entry.completion_prefix
            prefix = SequentialExecutionPrefix(
                (
                    previous,
                    CapturedFlowPrefix(
                        context,
                        prefix.endpoint.frame,
                        position,
                        last.target.header_position,
                        kernel=self.kernel,
                    ),
                )
            )
        return prefix

    def require_interval_effects(self, interval: SingleFlowPrefix) -> None:
        self.kernel.require_interval_frame(interval)
        if interval in self._closed_intervals:
            return
        for occurrence in self.effects.occurrences(self.source, interval):
            occurrence.require_closed(self)
        self._closed_intervals.add(interval)

    def require_namespace_write(self, node: ast.Attribute) -> None:
        operation = self.source.mutation_operation(node)
        if not isinstance(
            cast(CompactMutation, operation.event).target, CompactAttributeTarget
        ):
            raise ValueError(
                "Source attribute write requires an actual attribute target"
            )
        self._require_storage_operation(operation)

    def require_item_write(self, node: ast.Subscript | ast.AnnAssign) -> None:
        operation = self.source.mutation_operation(node)
        if not isinstance(
            cast(CompactMutation, operation.event).target, CompactItemTarget
        ):
            raise ValueError(
                "Source item store requires an actual evaluated item assignment"
            )
        self._require_storage_operation(operation)

    def require_class_creation(self, node: ast.ClassDef) -> None:
        self.class_entry(node).require_installed_result()

    def require_binding_write(self, node: ast.AST) -> None:
        operations = tuple(
            site
            for site in self.source.operations_by_node.get(node, ())
            if isinstance(site.event, CompactMutation)
            and isinstance(site.event.target, CompactLexicalBindingTargetABC)
        )
        if not operations:
            raise ValueError("Source binding has no actual lexical mutation")
        for operation in operations:
            self._require_binding_operation(operation)

    def require_import(self, node: ast.Import | ast.ImportFrom) -> None:
        operations = tuple(
            site
            for site in self.source.operations_by_node.get(node, ())
            if isinstance(site.event, CompactMutation)
            and isinstance(site.event.target, CompactImportTarget)
        )
        if not operations:
            raise ValueError("Source import has no actual import binding operation")
        for operation in operations:
            self.require_import_operation(operation)

    def require_return(self, node: ast.Return) -> None:
        operation = self.source.node_operation(node, CompactEvaluatedResult)
        result = cast(CompactEvaluatedResult, operation.event)
        if result.destination.use is not CompactValueDestinationKind.RETURNED:
            raise ValueError("Return requires its original value destination")
        if result.value_use is None:
            return
        self.kernel._read_use(
            result.value_use,
            self.context_for_owner(operation.owner),
            frozenset(),
        ).require_closed()


@dataclass(frozen=True, eq=False)
class SourceFunctionEntry(
    NamespaceCreationEvidenceABC,
    RecordedNamespace,
    SourceExecutionEntryABC,
):
    """Fresh local storage for one canonical original source invocation."""

    activation: SourceFunctionActivationABC

    source = AliasProperty[SourceProductFlowProjection]("activation.environment.source")
    initial = AliasProperty[InitialNativeIsland](
        "activation.environment.kernel.initial"
    )
    context = AliasProperty[CompactFlowContext]("activation.callee.context")

    @property
    def creator_frame(self) -> InitialNativeFrame:
        prefix = self.activation.callee.native_frame_prefix
        prefix.require_admitted(self.initial)
        return prefix.endpoint.frame

    @property
    def globals(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.creator_frame.globals

    @property
    def builtins(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        return self.creator_frame.builtins

    @cached_property
    def initial_entries(
        self,
    ) -> Mapping[NativeScalar, CapturedReferenceResolution]:
        """Derive exact entry values from the activation's original binding."""
        return self.activation.initial_entries

    @cached_property
    def frame(self) -> InitialNativeFrame:
        _ = self.initial_entries
        return InitialNativeFrame(self, self.globals, self.builtins)

    def prefix(
        self, position: CompactFlowPosition | None, kernel: CapturedReferenceKernel
    ) -> AdmittedExecutionPrefixABC:
        self.require_admitted(kernel.initial)
        return self.activation.activation_prefix(
            CapturedFlowPrefix(self.context, self.frame, position, kernel=kernel)
        )

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if initial is not self.initial:
            raise ValueError("Source function belongs to a foreign native admission")
        self.activation.require_canonical_activation()

    def require_external_noninterference(
        self, prefix: AdmittedExecutionPrefixABC
    ) -> None:
        self.activation.callee.execution.entry.require_external_noninterference(prefix)

    def require_native_behavior(self, authority: SourceActivationAuthorityABC) -> None:
        self.activation.callee.execution.entry.require_native_behavior(authority)

    def require_operation_completion(
        self, authority: SourceActivationAuthorityABC
    ) -> None:
        self.activation.callee.execution.entry.require_operation_completion(authority)

    def require_native_creator(self, execution: ExactNativeFunctionExecution) -> None:
        self.require_admitted(self.initial)
        if execution is not self.activation.callee.native_execution:
            raise ValueError("Native creator belongs to a different function body")

    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        return self.initial_entries.get(key)


@dataclass(eq=False)
class ActivationLocalEffectResolver(
    CompactMutationResolverABC[ContextualMutation, None]
):
    """Admit only effects retained by one activation's local frame."""

    entry: SourceExecutionEntryABC

    def require_retained(self, prefix: AdmittedExecutionPrefixABC) -> None:
        for occurrence in prefix.mutation_occurrences():
            if (
                occurrence.source.context is self.entry.context
                and occurrence.source.frame.locals is self.entry
            ):
                occurrence.source.require_definite(occurrence.mutation)
                occurrence.mutation.resolve(self, occurrence)

    def _binding_mutation_resolution(
        self,
        context: ContextualMutation,
        mutation: CompactMutation,
        name: str,
    ) -> None:
        if (
            context.source.frame.binding_namespace(context.source.context, name)
            is not self.entry
        ):
            raise ValueError("External function effect transport remains unproved")

    def _receiver_mutation_resolution(
        self,
        context: ContextualMutation,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> None:
        raise ValueError("Receiver effect transport remains unproved")


@dataclass(eq=False)
class SourceFunctionExecution(SourceExecutionABC, CompactBranchPredicateResolverABC):
    """One immediate source-function activation selected by its original call."""

    entry: SourceFunctionEntry

    @cached_property
    def initial_contents(self) -> CapturedEntryContents:
        return CapturedEntryContents(self.kernel, self.entry)

    @property
    def fast_local_bindings(self) -> tuple[NativeBindingTransfer, ...]:
        return self.entry.activation.entry_continuation.bindings

    @cached_property
    def completed_return(self) -> SourceFunctionReturn:
        completion = SourceCompletionResolver(self).completed_body(self.entry.context)
        if not isinstance(completion, SourceFunctionReturn):
            raise ValueError("Function body has no unique proved return")
        return completion

    @cached_property
    def completion_prefix(self) -> AdmittedExecutionPrefixABC:
        return self.require_terminal_evaluation(
            self.completed_return.completion_evaluation
        )

    @cached_property
    def completed_result(self) -> CapturedReferenceResolution:
        return self.completed_return.result(self.completion_prefix)

    @cached_property
    def completed_locals(self) -> NamespaceMemberInventory:
        """Derive the local references released by this exact returning frame."""
        return NamespaceMemberInventory(self.kernel, self.entry, self.completion_prefix)

    def require_frame_cleanup(self) -> None:
        """Prove final local decrements or their retention by the return value.

        The native return receipt and source join establish that ``completed_result``
        escapes the frame.  Every other live local reference is released when the
        activation ends and therefore retains its ordinary lifetime obligation.
        """
        returned = self.completed_result
        returned.require_closed()
        for name in self.completed_locals.names:
            value = self.completed_locals.require_member(name)
            value.require_closed()
            if value is returned or value.proves_same_object(returned):
                continue
            value.require_release_in(self.entry.frame)

    def proves_boolean(
        self,
        predicate_use: CompactCallableReferenceUse,
        expected: bool,
    ) -> bool:
        """Resolve one direct predicate from this activation's entry bindings."""
        flow = self.entry.context.flow
        binding = flow.initial_parameter_binding_for_predicate(predicate_use)
        if binding is None or binding.parameter.name not in self.entry.initial_entries:
            return False
        value = self.entry.initial_entries[binding.parameter.name]
        try:
            value.require_constant_contents(expected)
        except ValueError:
            return False
        return True

    def require_returned_parameter_identity(
        self, parameter_name: str
    ) -> CapturedReferenceResolution:
        """Prove every possible successful return retains one entry parameter."""
        self.entry.require_admitted(self.initial)
        self.entry.activation.callee.native_execution.mode.require_immediate_activation()
        flow = self.entry.context.flow
        if parameter_name not in self.entry.initial_entries:
            raise ValueError("Returned parameter is absent from this activation")
        binding = flow.require_returned_parameter_binding(parameter_name, self)
        if binding.parameter.name != parameter_name:
            raise ValueError("Returned parameter differs from this activation")
        return self.entry.initial_entries[parameter_name]

    def require_closed(self) -> None:
        _ = self.entry.frame
        _ = self.entry.activation.entry_continuation
        _ = self.completed_result
        ActivationLocalEffectResolver(self.entry).require_retained(
            self.completion_prefix
        )
        self.require_frame_cleanup()

    def resolve_fast_local_store(
        self, binding: NativeBindingTransfer
    ) -> tuple[str, CapturedReferenceResolution]:
        resolver = SourceCompletionResolver(self)
        matches = []
        for mutation in self.entry.context.flow.mutations:
            completion = resolver.resolve(mutation)
            if (
                isinstance(completion, SourceAssignmentStore)
                and completion.production.binding is binding
            ):
                matches.append(completion)
        if len(matches) != 1:
            raise ValueError(
                "Native fast-local store has no unique source assignment owner"
            )
        return matches[0]._fast_local_store_resolution(binding)

    def result(self) -> CapturedReferenceResolution:
        self.require_closed()
        return self.completed_result

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SourceFunctionEntry):
            raise TypeError("Function execution requires an actual function entry")


@dataclass(eq=False)
class SourceModuleExecution(SourceExecutionABC):
    """Fresh module execution with its declared loader and native entry work."""

    entry: SourceModuleEntryPremise

    @cached_property
    def initial_contents(self) -> SourceModuleEntryContents:
        return SourceModuleEntryContents(self)

    @classmethod
    def from_source(cls, source: SourceProductFlowProjection) -> SourceModuleExecution:
        """Activate the supplied original observations under the standard premise.

        Callers with another entry convention supply their premise to the
        constructor. Observation identity alone grants no execution/effect rights.
        """
        return cls(ImportedSourceModuleEntryPremise.from_source(source))

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SourceModuleEntryPremise):
            raise TypeError("Source execution requires an actual source entry premise")

    @classmethod
    def from_module(cls, module: ParsedModule) -> SourceModuleExecution:
        """Collect a standalone source task under the standard import premise."""
        return cls.from_source(source_product_flow_projection(module))
