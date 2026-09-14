"""Native raw-code execution evidence without executing analyzed modules."""

from __future__ import annotations

import ast
import marshal
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
)
from dataclasses import (
    dataclass,
    field,
    replace,
)
import dis
from enum import (
    Enum,
    IntEnum,
    StrEnum,
    auto,
)
from itertools import takewhile
from functools import (
    cached_property,
    lru_cache,
    partial,
)
import inspect
import sys
from operator import attrgetter
from types import (
    CodeType,
    FunctionType,
)
from typing import (
    ClassVar,
    Generic,
    Self,
    TYPE_CHECKING,
    TypeVar,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from .call_binding import CompactKeywordArgument
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
)
from .descriptor_algebra import AliasProperty
from .lexical_bindings import (
    CompactParameterKind,
    FunctionAnnotationVisitor,
    FunctionArgumentSource,
)
from .native_class_mro import NativeClassMroDeclaration
from .native_declarations import (
    NativeDeclaration,
    NativeScalar,
    NativeConstantContentsABC,
    NativeScalarValueABC,
    NativeTypeDeclaration,
)
from .scan_cache import ScanCache
from .source_geometry import SourceByteSpan
from .source_identity import (
    python_source_cache_signature,
    source_path_text,
)
from .value_expression import LiteralExpressionEffects
from .value_graph import DataclassGraphNode, DataclassGraphValue, StoredDataclassState

if TYPE_CHECKING:
    from .native_subscription import InertNativeArgumentWitness
    from .captured_reference import (
        CapturedReferenceResolution,
        NamespaceCreationEvidenceABC,
    )


class NativeFunctionExecutionMode(IntEnum):
    """Native raw-code flags, not the identity of a decorated binding."""

    ORDINARY = 0
    GENERATOR = inspect.CO_GENERATOR
    COROUTINE = inspect.CO_COROUTINE
    ASYNC_GENERATOR = inspect.CO_ASYNC_GENERATOR

    @classmethod
    def from_flags(cls, flags: int) -> Self:
        return cls(flags & sum(member.value for member in cls))


class NativeExecutionUnavailable(StrEnum):
    COMPILATION_REJECTED = "compilation_rejected"
    INCOMPLETE_SOURCE_RANGES = "incomplete_source_ranges"
    NO_EMITTED_CODE = "no_emitted_code"
    AMBIGUOUS_SOURCE_SPAN = "ambiguous_source_span"

    UNSUPPORTED_COMPILER = "unsupported_compiler"

    NO_OBSERVED_CREATION = "no_observed_creation"

    UNJOINED_FRAME_ORIGIN = "unjoined_frame_origin"

    UNSUPPORTED_PROLOGUE = "unsupported_prologue"


@dataclass(frozen=True)
class NativeCompilationIdentity:
    """Exact source and interpreter provenance of one native compilation."""

    file_path: str
    source_digest: str
    interpreter: tuple[str, str]


@dataclass(frozen=True)
class NativeFunctionExecution(ABC):
    """Compact declaration evidence; no AST or executable code is retained."""

    compilation: NativeCompilationIdentity
    source_span: SourceByteSpan

    def require_installation(self) -> NativeBindingTransfer:
        """Require conditional immediate raw-function storage, not source activation."""
        raise ValueError("Native raw function installation remains unproved")

    def require_creation(self) -> NativeCaptureSite:
        """Require observed raw MAKE_FUNCTION, not installed value or activation."""
        raise ValueError("Native raw function creation remains unproved")

    def require_applied_installation(self) -> NativeBindingTransfer:
        raise ValueError("Native applied-result installation remains unproved")

    def require_applications(self) -> tuple[NativeDefinitionApplication, ...]:
        raise ValueError("Native definition applications remain unproved")

    @property
    @abstractmethod
    def mode(self) -> NativeFunctionExecutionMode | None:
        raise NotImplementedError

    @property
    def violation(self) -> NativeExecutionUnavailable | None:
        return None


@dataclass(frozen=True)
class ExactNativeFunctionExecution(NativeFunctionExecution):
    """One emitted raw code object at the exact requested source span."""

    native_flags: int

    @property
    def mode(self) -> NativeFunctionExecutionMode:
        return NativeFunctionExecutionMode.from_flags(self.native_flags)


@dataclass(frozen=True)
class CreatedNativeFunctionExecution(ExactNativeFunctionExecution):
    """Raw code with its original observed native creation site.

    A source body can execute multiple times. This receipt establishes its
    compiler creation relation, not one runtime function, installed binding,
    creator activation, or captured builtin namespace.
    """

    creation: NativeCaptureSite

    def require_creation(self) -> NativeCaptureSite:
        return self.creation


@dataclass(frozen=True)
class InstalledNativeFunctionExecution(CreatedNativeFunctionExecution):
    """Original raw function followed immediately by its declared native store."""

    installation: NativeBindingTransfer

    def require_installation(self) -> NativeBindingTransfer:
        return self.installation


@dataclass(frozen=True)
class AppliedNativeFunctionExecution(CreatedNativeFunctionExecution):
    """An observed application chain stores a result distinct from raw creation."""

    applications: tuple[NativeDefinitionApplication, ...]
    applied_installation: NativeBindingTransfer

    def require_applied_installation(self) -> NativeBindingTransfer:
        return self.applied_installation

    def require_applications(self) -> tuple[NativeDefinitionApplication, ...]:
        return self.applications


@dataclass(frozen=True)
class OpenNativeFunctionExecution(NativeFunctionExecution):
    """The native compiler did not provide a unique positioned code object."""

    reason: NativeExecutionUnavailable

    @property
    def mode(self) -> None:
        return None

    @property
    def violation(self) -> NativeExecutionUnavailable:
        return self.reason


NativeResolutionT = TypeVar("NativeResolutionT")


class NativeFrameOriginResolverABC(ABC, Generic[NativeResolutionT]):
    @abstractmethod
    def _module_frame_origin_resolution(
        self, origin: ModuleNativeFrameOrigin
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _source_frame_origin_resolution(
        self, origin: SourceNativeFrameOrigin
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _open_frame_origin_resolution(
        self, origin: OpenNativeFrameOrigin
    ) -> NativeResolutionT:
        raise NotImplementedError


class NativeClassCaptureResolverABC(ABC, Generic[NativeResolutionT]):
    @abstractmethod
    def _exact_class_capture_resolution(
        self, capture: ExactNativeClassCapture
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _open_class_capture_resolution(
        self, capture: OpenNativeClassCapture
    ) -> NativeResolutionT:
        raise NotImplementedError


class NativeFrameOrigin(ABC):
    """Compiler container provenance, not an admitted execution activation."""

    compilation: NativeCompilationIdentity

    def is_body_of(self, execution: NativeFunctionExecution) -> bool:
        return False

    @abstractmethod
    def resolve(
        self, resolver: NativeFrameOriginResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        raise NotImplementedError


@dataclass(frozen=True)
class ModuleNativeFrameOrigin(NativeFrameOrigin):
    compilation: NativeCompilationIdentity

    def resolve(
        self, resolver: NativeFrameOriginResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._module_frame_origin_resolution(self)


@dataclass(frozen=True)
class SourceNativeFrameOrigin(NativeFrameOrigin):
    execution: ExactNativeFunctionExecution

    def is_body_of(self, execution: NativeFunctionExecution) -> bool:
        return self.execution is execution

    @property
    def compilation(self) -> NativeCompilationIdentity:
        return self.execution.compilation

    def resolve(
        self, resolver: NativeFrameOriginResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._source_frame_origin_resolution(self)


@dataclass(frozen=True)
class OpenNativeFrameOrigin(NativeFrameOrigin):
    """An actual container without a unique explicit source-body owner.

    Generated wrappers remain here unless a separate activation proof exists.
    Their code is not equated to a source parent by overlapping source ranges.
    """

    compilation: NativeCompilationIdentity
    reason: NativeExecutionUnavailable

    def resolve(
        self, resolver: NativeFrameOriginResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._open_frame_origin_resolution(self)


class NativeBindingTransferResolverABC(ABC, Generic[NativeResolutionT]):

    def _fast_local_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise ValueError("Native fast-local store requires its admitted function frame")

    def _deletion_resolution(self, binding: NativeBindingTransfer) -> NativeResolutionT:
        raise ValueError("Native deletion effects require their admitted source cut")

    @abstractmethod
    def _cell_creation_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _local_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _global_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _cell_store_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _local_ensure_resolution(
        self, binding: NativeBindingTransfer
    ) -> NativeResolutionT:
        raise NotImplementedError


class NativeValueResolverABC(ABC, Generic[NativeResolutionT]):

    def _subscription_native_value_resolution(
        self, value: NativeSubscriptionValue
    ) -> NativeResolutionT:
        return self._unproved_native_value_resolution(value)

    def _call_native_value_resolution(
        self, value: NativeCallValue
    ) -> NativeResolutionT:
        return self._unproved_native_value_resolution(value)

    def _function_native_value_resolution(
        self, value: NativeFunctionValue
    ) -> NativeResolutionT:
        return self._typed_native_value_resolution(value)

    def _attribute_native_value_resolution(
        self, value: NativeAttributeValue
    ) -> NativeResolutionT:
        return self._unproved_native_value_resolution(value)

    def _tuple_native_value_resolution(
        self, value: NativeTupleValue
    ) -> NativeResolutionT:
        return self._typed_native_value_resolution(value)

    def _list_native_value_resolution(
        self, value: NativeListValue
    ) -> NativeResolutionT:
        return self._typed_native_value_resolution(value)

    def _empty_dictionary_native_value_resolution(
        self, value: NativeEmptyDictionaryValue
    ) -> NativeResolutionT:
        return self._typed_native_value_resolution(value)

    def _annotation_namespace_native_value_resolution(
        self, value: NativeAnnotationNamespaceValue
    ) -> NativeResolutionT:
        return self._unproved_native_value_resolution(value)

    @abstractmethod
    def _unproved_native_value_resolution(
        self, value: NativeProducedValue
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _typed_native_value_resolution(
        self, value: NativeTypedValue
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _name_native_value_resolution(
        self, value: NativeNameValue
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _global_native_value_resolution(
        self, value: NativeGlobalValue
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _local_native_value_resolution(
        self, value: NativeLocalValue
    ) -> NativeResolutionT:
        raise NotImplementedError


class NativeCompletionResolverABC(ABC):
    """Completion obligations distinguish produced values from void transfers."""

    @abstractmethod
    def _native_value_completion(self, value: NativeProducedValue) -> None:
        raise NotImplementedError

    @abstractmethod
    def _native_stack_effect_completion(self, effect: NativeStackEffectABC) -> None:
        raise NotImplementedError


class NativeCallSlotABC(ABC):
    """An original CALL prefix operand: protocol NULL or an implicit argument."""

    instruction_offset: int

    @property
    @abstractmethod
    def implicit_arguments(self) -> tuple[NativeProducedValue, ...]:
        raise NotImplementedError

    @abstractmethod
    def split_call_prefix(
        self, order: NativeCallOperandOrder, operands: tuple[NativeCallSlotABC, ...]
    ) -> tuple[NativeProducedValue, NativeCallSlotABC]:
        raise NotImplementedError


@dataclass(frozen=True)
class NativeProducedValue(DataclassGraphNode, NativeCallSlotABC):
    """One original native production and its inputs, not runtime identity."""

    instruction_offset: int
    inputs: tuple[NativeProducedValue, ...]

    graph_children = AliasProperty["tuple[NativeProducedValue, ...]"]("inputs")

    source_span: SourceByteSpan | None = field(default=None, kw_only=True)

    def require_completion(self, resolver: NativeCompletionResolverABC) -> None:
        resolver._native_value_completion(self)

    @property
    def implicit_arguments(self) -> tuple[NativeProducedValue, ...]:
        return (self,)

    def split_call_prefix(
        self, order: NativeCallOperandOrder, operands: tuple[NativeCallSlotABC, ...]
    ) -> tuple[NativeProducedValue, NativeProducedValue]:
        # Both supported ABIs place an implicit object after its callable.
        # The enum selects the dispatch operand using its NULL layout; neither
        # position can then contain a protocol marker in this object form.
        if any(not isinstance(value, NativeProducedValue) for value in operands):
            raise ValueError("Native implicit call prefix requires Python operands")
        callee, argument = operands
        return callee, argument

    def productions(self) -> tuple[NativeProducedValue, ...]:
        """Original reachable inputs in a forward native operand graph."""
        values = tuple(cast(NativeProducedValue, node) for node in self.graph_nodes())
        for value in values:
            if any(
                dependency.instruction_offset >= value.instruction_offset
                for dependency in value.inputs
            ):
                raise ValueError(
                    "Native operands require strictly preceding productions"
                )
        return values

    def require_scalar_store_value(self) -> NativeConstantValue:
        """A produced type or read alone does not establish scalar contents."""
        raise ValueError("Native scalar production remains unproved")

    def require_literal_contents(self, expected: object) -> None:
        """A transfer alone does not prove the contents of a source literal."""
        raise ValueError("Native literal construction contents remain unproved")

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._unproved_native_value_resolution(self)


@dataclass(frozen=True)
class NativeTypedValue(NativeProducedValue, NativeScalarValueABC):
    declaration: NativeTypeDeclaration

    native_type = AliasProperty[type]("declaration.declaration")

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._typed_native_value_resolution(self)


@dataclass(frozen=True, eq=False)
class NativeFunctionValue(DataclassGraphValue, NativeTypedValue):
    """Original raw creation operands; this is not source activation or identity."""

    declaration: ClassVar[NativeTypeDeclaration] = NativeTypeDeclaration(FunctionType)

    @property
    def creation(self) -> NativeFunctionValue:
        return self

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._function_native_value_resolution(self)


@dataclass(frozen=True, eq=False)
class NativeFunctionAttributeValue(NativeFunctionValue):
    """An attribute attachment retains its original function creation."""

    attribute_flag: int

    @property
    def function(self) -> NativeFunctionValue:
        return cast(NativeFunctionValue, self.inputs[-1])

    creation = AliasProperty[NativeFunctionValue]("function.creation")


@dataclass(frozen=True, eq=False)
class NativeCallValue(DataclassGraphValue, NativeProducedValue):
    """Original invocation with ordered implicit and explicit Python arguments."""

    argument_slot: NativeCallSlotABC

    @property
    def callee(self) -> NativeProducedValue:
        return self.inputs[0]

    @property
    def arguments(self) -> tuple[NativeProducedValue, ...]:
        return self.inputs[1:]

    @property
    def positional_arguments(self) -> tuple[NativeProducedValue, ...]:
        return self.arguments

    @property
    def keyword_arguments(
        self,
    ) -> tuple[CompactKeywordArgument[NativeProducedValue], ...]:
        return ()

    def require_argument_shape(
        self, positional_count: int, keyword_names: tuple[str | None, ...]
    ) -> None:
        """Compare source argument shape without supplying operand identity."""
        if positional_count != len(self.positional_arguments):
            raise ValueError(
                "Native call has a different original source argument count"
            )
        if keyword_names != tuple(argument.name for argument in self.keyword_arguments):
            raise ValueError("Native call has different original source keyword names")

    def require_definition_argument(self) -> NativeProducedValue:
        """Require the sole implicit argument of a compiler definition application."""
        self.require_argument_shape(1, ())
        if self.argument_slot is not self.arguments[0]:
            raise ValueError("Native application requires its single implicit argument")
        return self.arguments[0]

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._call_native_value_resolution(self)


@dataclass(frozen=True, eq=False)
class NativeKeywordCallValue(NativeCallValue):
    """Original keyword metadata and values, independent of source admission."""

    keyword_names: NativeConstantValue = field(kw_only=True)

    def __post_init__(self) -> None:
        self.names

    @property
    def graph_children(self) -> tuple[NativeProducedValue, ...]:
        return (*super().graph_children, self.keyword_names)

    @property
    def names(self) -> tuple[str, ...]:
        if not isinstance(self.keyword_names, NativeConstantValue):
            raise ValueError("Native keyword names require original constant metadata")
        names = self.keyword_names.value
        if type(names) is not tuple or any(type(name) is not str for name in names):
            raise ValueError("Native keyword names require an exact string tuple")
        if len(set(names)) != len(names) or len(names) > (
            len(self.arguments) - len(self.argument_slot.implicit_arguments)
        ):
            raise ValueError("Native keyword names do not match explicit arguments")
        if self.keyword_names.instruction_offset >= self.instruction_offset:
            raise ValueError("Native keyword metadata must precede its invocation")
        return names

    @property
    def positional_arguments(self) -> tuple[NativeProducedValue, ...]:
        return self.arguments[: len(self.arguments) - len(self.names)]

    @property
    def keyword_arguments(
        self,
    ) -> tuple[CompactKeywordArgument[NativeProducedValue], ...]:
        names = self.names
        return tuple(
            CompactKeywordArgument(name, operand)
            for name, operand in zip(
                names, self.arguments[len(self.arguments) - len(names) :], strict=True
            )
        )


@dataclass(frozen=True)
class NativeSequenceValue(NativeTypedValue):
    """Ordered inputs of a native sequence builder, not its runtime identity."""

    declaration: ClassVar[NativeTypeDeclaration]

    @property
    @abstractmethod
    def declaration(self) -> NativeTypeDeclaration:
        """The concrete builder supplies its native sequence declaration."""
        raise NotImplementedError

    def require_literal_contents(self, expected: object) -> None:
        if type(expected) is not self.native_type or len(expected) != len(self.inputs):
            raise ValueError("Native sequence differs from its source literal shape")
        for operand, item in zip(self.inputs, expected, strict=True):
            operand.require_literal_contents(item)

    @classmethod
    def capture(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        if type(instruction.arg) is not int or instruction.arg < 0:
            raise ValueError("Native sequence has no exact nonnegative input count")
        stack.stack.append(
            stack.emit(cls(instruction.offset, stack.pop(instruction.arg)), instruction)
        )


class NativeListValue(NativeSequenceValue):
    """Original BUILD_LIST operands; later mutation and source effects stay open."""

    declaration = NativeTypeDeclaration(list)

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._list_native_value_resolution(self)


class NativeListExtensionValue(NativeListValue):
    """Original in-place list extension with its receiver and iterable operand."""

    @classmethod
    def capture(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        depth = instruction.arg
        if type(depth) is not int or depth < 1:
            raise ValueError("Native list extension has no exact positive depth")
        (iterable,) = stack.pop(1)
        if depth > len(stack.stack):
            raise ValueError("Native list extension has no original receiver")
        receiver = stack.stack[-depth]
        if not isinstance(receiver, NativeListValue) or any(
            value is receiver
            for index, value in enumerate(stack.stack)
            if index != len(stack.stack) - depth
        ):
            raise ValueError("Native list extension requires an unaliased list builder")
        stack.stack[-depth] = stack.emit(
            cls(instruction.offset, (receiver, iterable)), instruction
        )

    def require_literal_contents(self, expected: object) -> None:
        receiver, iterable = self.inputs
        if type(expected) is not list or not isinstance(iterable, NativeConstantValue):
            raise ValueError("Native literal list extension has no constant iterable")
        items = iterable.value
        if type(items) is not tuple or len(items) > len(expected):
            raise ValueError("Native literal list extension has incompatible items")
        boundary = len(expected) - len(items)
        receiver.require_literal_contents(expected[:boundary])
        iterable.require_constant_contents(tuple(expected[boundary:]))


@dataclass(frozen=True)
class NativeTupleValue(NativeSequenceValue):
    """Original BUILD_TUPLE and its ordered operands, not a constant pool value."""

    declaration: ClassVar[NativeTypeDeclaration] = NativeTypeDeclaration(tuple)

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._tuple_native_value_resolution(self)


@dataclass(frozen=True)
class NativeConstantValue(NativeTypedValue):
    """Exact contents retained at their original native constant production."""

    declaration: ClassVar[NativeTypeDeclaration]
    value: object

    def _native_constant_value(self) -> object:
        return self.value

    def require_literal_contents(self, expected: object) -> None:
        self.require_constant_contents(expected)

    def require_scalar_store_value(self) -> NativeConstantValue:
        self.require_native_scalar()
        return self

    @property
    def declaration(self) -> NativeTypeDeclaration:
        return NativeTypeDeclaration(type(self.value))

    def _native_scalar_value(self) -> object:
        return self.value


@dataclass(frozen=True)
class NativeEmptyDictionaryValue(NativeTypedValue):
    """Fresh exact empty-map production; source activation remains separate."""

    declaration: ClassVar[NativeTypeDeclaration] = NativeTypeDeclaration(dict)

    inputs: ClassVar[tuple[NativeProducedValue, ...]] = ()

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._empty_dictionary_native_value_resolution(self)


@dataclass(frozen=True)
class NativeAnnotationNamespaceValue(NativeTypedValue):
    """SETUP_ANNOTATIONS observes existing storage or creates an exact dictionary.

    Freshness requires independent absence at this original frame's entry.
    """

    namespace_name: ClassVar[str] = "__annotations__"
    declaration: ClassVar[NativeTypeDeclaration] = NativeTypeDeclaration(dict)
    inputs: ClassVar[tuple[NativeProducedValue, ...]] = ()

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._annotation_namespace_native_value_resolution(self)


class NativeStackEffectABC(NativeProducedValue, ABC):
    """Original consumed operands; this observation supplies no Python result."""

    def require_completion(self, resolver: NativeCompletionResolverABC) -> None:
        resolver._native_stack_effect_completion(self)

    @property
    @abstractmethod
    def operand_count(self) -> int:
        raise NotImplementedError

    @classmethod
    def capture(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        stack.emit(cls(instruction.offset, stack.pop(cls.operand_count)), instruction)


NativeEffectT = TypeVar("NativeEffectT", bound=NativeStackEffectABC)


class NativeDiscardValue(NativeStackEffectABC):
    """Original operand release; source lifetime/effects remain unproved."""

    operand_count: ClassVar[int] = 1


class NativeItemStoreOperand(IntEnum):
    """STORE_SUBSCR's original operand roles in native stack order."""

    VALUE = 0
    RECEIVER = 1
    KEY = 2

    def select(self, effect: NativeItemStoreValue) -> NativeProducedValue:
        if not isinstance(effect, NativeItemStoreValue) or len(effect.inputs) != len(
            type(self)
        ):
            raise ValueError("Item operand requires its complete original transfer")
        return effect.inputs[self.value]


class NativeItemStoreValue(NativeStackEffectABC):
    """STORE_SUBSCR consumes value, receiver and key in original stack order.

    Custom item protocols and release effects require independent source proof.
    This transfer does not replace the receiver by an analyser-created mapping.
    """

    operand_count: ClassVar[int] = len(NativeItemStoreOperand)


@dataclass(frozen=True)
class NativeClassBuilderValue(NativeProducedValue):
    """Original builtins-only builder load; its runtime identity remains unproved."""

    inputs: ClassVar[tuple[NativeProducedValue, ...]] = ()


@dataclass(frozen=True)
class NativeReadValue(NativeProducedValue):
    operation: NativePrimitiveOperation
    name: str
    operand_index: int | None

    @classmethod
    def place_read(
        cls,
        stack: NativeOperandStack,
        value: NativeProducedValue,
        instruction: dis.Instruction,
    ) -> None:
        stack.stack.append(value)

    @classmethod
    def read_inputs(
        cls, stack: NativeOperandStack, instruction: dis.Instruction
    ) -> tuple[NativeProducedValue, ...]:
        return stack.pop(1 - dis.stack_effect(instruction.opcode, instruction.arg))

    @classmethod
    def capture(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        if type(instruction.argval) is not str:
            raise ValueError("Native prologue read has no exact name operand")
        inputs = cls.read_inputs(stack, instruction)
        value = cls(
            instruction.offset, inputs, operation, instruction.argval, instruction.arg
        )
        cls.place_read(stack, stack.emit(value, instruction), instruction)


class NativeNameValue(NativeReadValue):
    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._name_native_value_resolution(self)


class NativeImportValue(NativeReadValue):
    """Original module request consumes level and from-list, not a name lookup.

    The backend stack effect supplies input arity. Importer identity, effects
    and the returned object's identity remain source admission obligations.
    """

    def require_request(
        self, name: str, level: int, from_list: tuple[str, ...] | None
    ) -> None:
        if type(self.name) is not str or self.name != name or len(self.inputs) != 2:
            raise ValueError("Native import differs from its original source request")
        for value, expected in zip(self.inputs, (level, from_list), strict=True):
            if not isinstance(value, NativeConstantContentsABC):
                raise ValueError(
                    "Native import request has no original constant inputs"
                )
            value.require_constant_contents(expected)


class NativeImportMemberValue(NativeReadValue):
    """IMPORT_FROM retains its original module operand below the member result."""

    @classmethod
    def read_inputs(
        cls, stack: NativeOperandStack, instruction: dis.Instruction
    ) -> tuple[NativeProducedValue, ...]:
        (module,) = stack.pop(1)
        stack.stack.append(module)
        return (module,)


class NativeAttributeValue(NativeReadValue):
    """One receiver consumed by an ordinary attribute read, not method binding."""

    @classmethod
    def read_inputs(
        cls, stack: NativeOperandStack, instruction: dis.Instruction
    ) -> tuple[NativeProducedValue, ...]:
        if dis.stack_effect(instruction.opcode, instruction.arg) != 0:
            raise ValueError("Native attribute method-call binding remains unproved")
        return stack.pop(1)

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._attribute_native_value_resolution(self)


class NativeSubscriptionValue(NativeProducedValue):
    """Original receiver and key production; callbacks and result stay unproved."""

    @classmethod
    def capture(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        stack.stack.append(
            stack.emit(cls(instruction.offset, stack.pop(2)), instruction)
        )

    @classmethod
    def capture_binary(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        if instruction.arg != stack.backend.binary_subscription_argument:
            raise ValueError("Native binary operation is not subscription")
        cls.capture(stack, operation, instruction)

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._subscription_native_value_resolution(self)


class NativeGlobalValue(NativeReadValue):

    @classmethod
    def place_read(
        cls,
        stack: NativeOperandStack,
        value: NativeProducedValue,
        instruction: dis.Instruction,
    ) -> None:
        outputs = dis.stack_effect(instruction.opcode, instruction.arg)
        if outputs == 1:
            super().place_read(stack, value, instruction)
        elif outputs == 2:
            stack.stack.extend(
                stack.backend.call_operand_order.compose(
                    value, NativeCallMarker.from_instruction(instruction)
                )
            )
        else:
            raise ValueError("Native global load has an unsupported output arity")

    @classmethod
    def read_inputs(
        cls, stack: NativeOperandStack, instruction: dis.Instruction
    ) -> tuple[NativeProducedValue, ...]:
        return ()

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._global_native_value_resolution(self)


@dataclass(frozen=True)
class NativeLocalValue(NativeReadValue):
    """A fast-local read; its value and type require the actual frame's evidence."""

    @classmethod
    def capture_many(
        cls,
        stack: NativeOperandStack,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:
        names = instruction.argval
        if (
            type(names) is not tuple
            or any(type(name) is not str for name in names)
            or len(names) != dis.stack_effect(instruction.opcode, instruction.arg)
            or stack.code is None
        ):
            raise ValueError("Native local read group requires its original decoded slots")
        for name in names:
            index = stack.code.co_varnames.index(name)
            value = cls(instruction.offset, (), operation, name, index)
            cls.place_read(stack, stack.emit(value, instruction), instruction)

    def resolve(
        self, resolver: NativeValueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._local_native_value_resolution(self)


@dataclass(frozen=True)
class NativeBindingTransfer(StoredDataclassState):
    """One native write/ensure operation, not a namespace/value admission."""

    operation: NativePrimitiveOperation
    name: str
    instruction_offset: int
    operand_index: int | None

    value: NativeProducedValue | None = None

    source_span: SourceByteSpan | None = field(default=None, kw_only=True)

    def resolve(
        self, resolver: NativeBindingTransferResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return self.operation.resolve_binding(self, resolver)

    def __reduce__(self):
        # Original native inputs precede their users. Pickle memoises them before
        # serialising the binding's fields, retaining sharing without recursion
        # through a long call/decorator chain or persisting another value index.
        dependencies = (
            ()
            if self.value is None
            else tuple(
                sorted(self.value.productions(), key=attrgetter("instruction_offset"))
            )
        )
        return type(self)._restore, (dependencies, self.__getstate__())

    @classmethod
    def _restore(
        cls, dependencies: tuple[NativeProducedValue, ...], state: dict[str, object]
    ) -> NativeBindingTransfer:
        return cls(**state)

    def require_observation(self, observation: NativeBindingTransfer) -> None:
        """An operand-free observation may refer to this original store.

        Instruction metadata must agree. A second produced operand, even an
        equal copy, cannot replace the original operand authority.
        """
        if self is observation:
            return
        if replace(self, value=None) != replace(observation, value=None):
            raise ValueError("Store observations describe a different native transfer")
        if observation.value is not None and observation.value is not self.value:
            raise ValueError("Store observation does not retain its original operand")


class NativeClassPrologueResolverABC(ABC, Generic[NativeResolutionT]):
    @abstractmethod
    def _exact_class_prologue_resolution(
        self, prologue: ExactNativeClassPrologue
    ) -> NativeResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _open_class_prologue_resolution(
        self, prologue: OpenNativeClassPrologue
    ) -> NativeResolutionT:
        raise NotImplementedError


class NativeValueInventoryABC(StoredDataclassState, ABC):
    """Original operand productions retained by one native instruction walk."""

    @property
    @abstractmethod
    def values(self) -> tuple[NativeProducedValue, ...]:
        raise NotImplementedError

    @cached_property
    def _productions_by_offset(self) -> dict[int, NativeProducedValue]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.values, lambda value: value.instruction_offset
        )

    def production_at(self, instruction_offset: int) -> NativeProducedValue:
        try:
            return self._productions_by_offset[instruction_offset]
        except KeyError as error:
            raise ValueError(
                "Native value has no unique original production"
            ) from error

    @cached_property
    def _production_identities(self) -> frozenset[int]:
        """An instruction may produce several distinct original values."""
        return frozenset(id(value) for value in self.values)

    def require_value(self, value: NativeProducedValue) -> None:
        if id(value) not in self._production_identities:
            raise ValueError("Native value requires its original production")


class NativeClassPrologue(ABC):

    def require_prior_production(self, instruction_offset: int) -> NativeProducedValue:
        raise ValueError("Native prologue production remains unproved")

    def require_body_start(self) -> int:
        raise ValueError("Native prologue has no proved body boundary")

    def require_fresh_cell(
        self, name: str, operand_index: int | None, offset: int
    ) -> None:
        raise ValueError("Native prologue cell remains unproved")

    def require_value(self, value: NativeProducedValue) -> None:
        raise ValueError("Native prologue value remains unproved")

    @abstractmethod
    def require_fresh_cell_store(self, binding: NativeBindingTransfer) -> None:
        """Require an actual earlier fresh-cell creation for this native store."""
        raise NotImplementedError

    @abstractmethod
    def resolve(
        self, resolver: NativeClassPrologueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactNativeClassPrologue(NativeValueInventoryABC, NativeClassPrologue):
    """Contiguous compiler primitives before the first non-prologue instruction.

    Owned by an exact class capture. This is neither activation nor proof of
    callback-free lookup, prepared storage or namespace values. MAKE_CELL
    receipts retain conditional own-frame cell creation, not frame activation.
    The native boundary may be tail/return code when source statements emit none.
    """

    bindings: tuple[NativeBindingTransfer, ...]
    body_instruction_offset: int

    values: tuple[NativeProducedValue, ...] = ()

    def require_prior_production(self, instruction_offset: int) -> NativeProducedValue:
        if (
            instruction_offset >= self.body_instruction_offset
            or instruction_offset not in self._productions_by_offset
        ):
            raise ValueError("Native prologue has no unique prior production")
        return self._productions_by_offset[instruction_offset]

    def require_body_start(self) -> int:
        return self.body_instruction_offset

    def require_fresh_cell(
        self, name: str, operand_index: int | None, offset: int
    ) -> None:
        creations = tuple(
            candidate
            for candidate in self.bindings
            if candidate.operation is NativePrimitiveOperation.MAKE_CELL
            and candidate.operand_index == operand_index
            and candidate.name == name
            and candidate.instruction_offset < offset
        )
        if type(operand_index) is not int or len(creations) != 1:
            raise ValueError(
                "Cell operand has no unique prior creation in this native frame"
            )

    def require_fresh_cell_store(self, binding: NativeBindingTransfer) -> None:
        """Require the actual store's unique prior cell creation in this raw frame.

        Original compiler MAKE_CELL allocates fresh cell storage. Its native slot
        operand, unlike the display name alone, joins STORE_DEREF in the same code
        object. This conditional execution fact is not an activation admission.
        """
        if not (
            sum(candidate is binding for candidate in self.bindings) == 1
            and binding.operation is NativePrimitiveOperation.STORE_DEREF
            and type(binding.operand_index) is int
        ):
            raise ValueError(
                "Fresh-cell proof requires this prologue's actual cell store"
            )
        self.require_fresh_cell(
            binding.name, binding.operand_index, binding.instruction_offset
        )

    def resolve(
        self, resolver: NativeClassPrologueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._exact_class_prologue_resolution(self)


@dataclass(frozen=True)
class OpenNativeClassPrologue(NativeClassPrologue):
    reason: NativeExecutionUnavailable

    def require_fresh_cell_store(self, binding: NativeBindingTransfer) -> None:
        raise ValueError(
            f"Native cell creation remains unresolved: {self.reason.value}"
        )

    def resolve(
        self, resolver: NativeClassPrologueResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._open_class_prologue_resolution(self)


@dataclass(frozen=True)
class NativeCallMarker(NativeCallSlotABC):
    """Original protocol NULL, which is not a produced Python value."""

    instruction_offset: int
    source_span: SourceByteSpan | None

    implicit_arguments: ClassVar[tuple[NativeProducedValue, ...]] = ()

    def split_call_prefix(
        self, order: NativeCallOperandOrder, operands: tuple[NativeCallSlotABC, ...]
    ) -> tuple[NativeProducedValue, NativeCallMarker]:
        callee = operands[order.callee_index]
        if not isinstance(callee, NativeProducedValue):
            raise ValueError("Native call prefix differs from the backend protocol")
        return callee, self

    @classmethod
    def from_instruction(cls, instruction: dis.Instruction) -> NativeCallMarker:
        return cls(instruction.offset, NativeInstructionSite.span_for(instruction))


class NativeCallOperandOrder(Enum):
    """Compiler-owned NULL layout; operand declarations own the object form."""

    NULL_CALLEE = (1, 0)
    CALLEE_NULL = (0, 1)

    def compose(
        self, callee: NativeProducedValue, marker: NativeCallMarker
    ) -> tuple[NativeProducedValue | NativeCallMarker, ...]:
        roles = (callee, marker)
        return roles[self.callee_index], roles[self.marker_index]

    def __init__(self, callee_index: int, marker_index: int) -> None:
        self.callee_index = callee_index
        self.marker_index = marker_index

    def split(
        self, operands: tuple[NativeCallSlotABC, ...]
    ) -> tuple[NativeProducedValue, NativeCallSlotABC]:
        if len(operands) != 2:
            raise ValueError("Native call requires its exact two-slot prefix")
        return operands[self.marker_index].split_call_prefix(self, operands)


@dataclass
class NativeOperandStack:
    """Transient operand projection shared by native prefix and entry-store observations.

    Each production is retained once. Stores and dependent productions refer
    to it; no source namespace or analyzer runtime object is reconstructed.
    A typed production does not prove its inputs can be evaluated.
    """

    stack: list[NativeProducedValue | NativeCallMarker] = field(default_factory=list)
    values: list[NativeProducedValue] = field(default_factory=list)
    bindings: list[NativeBindingTransfer] = field(default_factory=list)

    returned: NativeProducedValue | None = None
    return_offset: int | None = None

    expected_operation: NativePrimitiveOperation | None = None

    call_prelude: list[dis.Instruction] = field(default_factory=list)
    code: CodeType | None = field(default=None, kw_only=True)
    call_factory: Callable[
        [int, tuple[NativeProducedValue, ...], NativeCallSlotABC], NativeCallValue
    ] = field(default=NativeCallValue, init=False)

    def capture_instruction(
        self,
        instruction: dis.Instruction,
        operation: NativePrimitiveOperation | None,
        creation_context: NativeCodeObservation | None = None,
    ) -> None:
        if operation is not None:
            operation.capture(self, instruction)
        elif creation_context is not None:
            self.backend.capture_function_operand(self, creation_context, instruction)
        else:
            raise ValueError("Native operand has no admitted instruction observation")

    def pop(self, count: int) -> tuple[NativeProducedValue, ...]:
        operands = self.take(count)
        if any(not isinstance(value, NativeProducedValue) for value in operands):
            raise ValueError("A call protocol marker is not a Python operand")
        return cast(tuple[NativeProducedValue, ...], operands)

    def invoke(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.backend.require_invocation(self.call_prelude, instruction)
        arguments = self.pop(instruction.arg)
        callee, argument_slot = self.backend.call_operand_order.split(self.take(2))
        if argument_slot.instruction_offset >= instruction.offset:
            raise ValueError("Native call argument slot must precede its invocation")
        self.stack.append(
            self.emit(
                self.call_factory(
                    instruction.offset,
                    (callee, *argument_slot.implicit_arguments, *arguments),
                    argument_slot,
                ),
                instruction,
            )
        )
        self.call_prelude.clear()
        self.call_factory = NativeCallValue

    def prepare_keyword_names(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        if (
            self.code is None
            or type(instruction.arg) is not int
            or not 0 <= instruction.arg < len(self.code.co_consts)
        ):
            raise ValueError(
                "Native keyword metadata requires its original code constant"
            )
        names = self.emit(
            NativeConstantValue(
                instruction.offset, (), self.code.co_consts[instruction.arg]
            ),
            instruction,
        )
        self.call_factory = partial(NativeKeywordCallValue, keyword_names=names)
        self.expected_operation = NativePrimitiveOperation.PRECALL

    def invoke_keywords(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        (names,) = self.pop(1)
        if not isinstance(names, NativeConstantValue):
            raise ValueError(
                "Native keyword metadata requires its original constant tuple"
            )
        self.call_factory = partial(NativeKeywordCallValue, keyword_names=names)
        self.invoke(operation, instruction)

    def prepare_call(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.call_prelude.append(instruction)
        self.expected_operation = NativePrimitiveOperation.CALL

    def push_call_marker(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.stack.append(NativeCallMarker.from_instruction(instruction))

    @cached_property
    def backend(self) -> NativeCreationBackend:
        return NativeCreationBackend.current()

    def class_builder(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.stack.append(
            self.emit(NativeClassBuilderValue(instruction.offset), instruction)
        )

    def dictionary_value(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        if type(instruction.arg) is not int or instruction.arg < 0:
            raise ValueError("Native dictionary has no exact nonnegative input count")
        self.stack.append(
            self.typed(instruction, dict, 2 * instruction.arg)
            if instruction.arg
            else self.emit(NativeEmptyDictionaryValue(instruction.offset), instruction)
        )

    def constant_key_dictionary(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        if type(instruction.arg) is not int or instruction.arg < 0:
            raise ValueError("Native dictionary has no exact nonnegative input count")
        inputs = self.pop(instruction.arg + 1)
        keys = inputs[-1]
        if (
            not isinstance(keys, NativeConstantValue)
            or type(keys.value) is not tuple
            or len(keys.value) != instruction.arg
        ):
            raise ValueError(
                "Native dictionary requires its original constant key tuple"
            )
        self.stack.append(
            self.emit(
                NativeTypedValue(
                    instruction.offset, inputs, NativeTypeDeclaration(dict)
                ),
                instruction,
            )
        )

    def return_value(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        (value,) = self.pop(1)
        if self.stack or self.returned is not None:
            raise ValueError(
                "Native return requires one result and no remaining operands"
            )
        self.returned = value
        self.return_offset = instruction.offset

    def return_constant(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.constant(operation, instruction)
        self.return_value(operation, instruction)

    def take(self, count: int) -> tuple[NativeProducedValue | NativeCallMarker, ...]:
        if count < 0 or count > len(self.stack):
            raise ValueError("Native prologue operand stack remains unproved")
        if count == 0:
            return ()
        result = tuple(self.stack[-count:])
        del self.stack[-count:]
        return result

    def emit(
        self, value: NativeProducedValue, instruction: dis.Instruction
    ) -> NativeProducedValue:
        value = replace(value, source_span=NativeInstructionSite.span_for(instruction))
        self.values.append(value)
        return value

    def typed(
        self, instruction: dis.Instruction, native_type: type, count: int = 0
    ) -> NativeProducedValue:
        return self.emit(
            NativeTypedValue(
                instruction.offset, self.pop(count), NativeTypeDeclaration(native_type)
            ),
            instruction,
        )

    def ignore(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        pass

    def bind(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.bindings.extend(operation.bindings(instruction))

    def store(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.bindings.extend(operation.bindings(instruction, self.pop(1)[0]))

    def constant(
        self,
        operation: NativePrimitiveOperation,
        instruction: dis.Instruction,
    ) -> None:

        value = instruction.argval
        production = (
            self.emit(NativeConstantValue(instruction.offset, (), value), instruction)
            if NativeConstantValue.supports_constant(value)
            else self.typed(instruction, type(value))
        )
        self.stack.append(production)

    def locals_value(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        self.stack.append(self.typed(instruction, dict))

    def annotations(
        self, operation: NativePrimitiveOperation, instruction: dis.Instruction
    ) -> None:
        value = self.emit(
            NativeAnnotationNamespaceValue(instruction.offset), instruction
        )
        self.bindings.extend(operation.bindings(instruction, value))

    def complete(self, offset: int) -> ExactNativeClassPrologue:
        return ExactNativeClassPrologue(
            tuple(self.bindings), offset, tuple(self.values)
        )


class NativePrimitiveOperation(Enum):
    """Native primitives own both operand production and storage projection.

    These are original compiler transfers, not effect-free execution claims.
    SETUP_ANNOTATIONS' implicit key belongs to that primitive alone.
    """

    RESUME = auto()
    NOP = auto()
    POP_TOP = (auto(), NativeDiscardValue.capture)
    STORE_SUBSCR = (auto(), NativeItemStoreValue.capture)
    PUSH_NULL = (auto(), NativeOperandStack.push_call_marker)
    PRECALL = (auto(), NativeOperandStack.prepare_call)
    CALL = (auto(), NativeOperandStack.invoke)
    KW_NAMES = (auto(), NativeOperandStack.prepare_keyword_names)
    CALL_KW = (auto(), NativeOperandStack.invoke_keywords)
    RETURN_VALUE = (
        auto(),
        NativeOperandStack.return_value,
        None,
        False,
        attrgetter("argval"),
        True,
    )
    RETURN_CONST = (
        auto(),
        NativeOperandStack.return_constant,
        None,
        False,
        attrgetter("argval"),
        True,
    )
    COPY_FREE_VARS = (auto(), NativeOperandStack.ignore, None, True)
    MAKE_CELL = (
        auto(),
        NativeOperandStack.bind,
        lambda resolver, binding: resolver._cell_creation_resolution(binding),
        True,
    )
    IMPORT_NAME = (auto(), NativeImportValue.capture)
    IMPORT_FROM = (auto(), NativeImportMemberValue.capture)
    LOAD_NAME = (auto(), NativeNameValue.capture)
    LOAD_ATTR = (auto(), NativeAttributeValue.capture)
    LOAD_GLOBAL = (auto(), NativeGlobalValue.capture)
    LOAD_CONST = (auto(), NativeOperandStack.constant)
    LOAD_SMALL_INT = (auto(), NativeOperandStack.constant)
    LOAD_FAST_BORROW = (auto(), NativeLocalValue.capture)
    BUILD_TUPLE = (auto(), NativeTupleValue.capture)
    BUILD_LIST = (auto(), NativeListValue.capture)
    LIST_EXTEND = (auto(), NativeListExtensionValue.capture)
    LOAD_BUILD_CLASS = (auto(), NativeOperandStack.class_builder)
    BUILD_MAP = (auto(), NativeOperandStack.dictionary_value)
    BUILD_CONST_KEY_MAP = (auto(), NativeOperandStack.constant_key_dictionary)
    LOAD_LOCALS = (auto(), NativeOperandStack.locals_value)
    LOAD_DEREF = (auto(), NativeReadValue.capture)
    LOAD_CLASSDEREF = (auto(), NativeReadValue.capture)
    LOAD_FROM_DICT_OR_DEREF = (auto(), NativeReadValue.capture)
    STORE_NAME = (
        auto(),
        NativeOperandStack.store,
        lambda resolver, binding: resolver._local_store_resolution(binding),
    )
    STORE_GLOBAL = (
        auto(),
        NativeOperandStack.store,
        lambda resolver, binding: resolver._global_store_resolution(binding),
    )
    STORE_DEREF = (
        auto(),
        NativeOperandStack.store,
        lambda resolver, binding: resolver._cell_store_resolution(binding),
    )
    DELETE_NAME = (
        auto(),
        NativeOperandStack.bind,
        lambda resolver, binding: resolver._deletion_resolution(binding),
        False,
        attrgetter("argval"),
        False,
        True,
    )
    DELETE_GLOBAL = (
        auto(),
        NativeOperandStack.bind,
        lambda resolver, binding: resolver._deletion_resolution(binding),
        False,
        attrgetter("argval"),
        False,
        True,
    )
    SETUP_ANNOTATIONS = (
        auto(),
        NativeOperandStack.annotations,
        lambda resolver, binding: resolver._local_ensure_resolution(binding),
        False,
        lambda instruction: NativeAnnotationNamespaceValue.namespace_name,
    )

    LOAD_FAST = (auto(), NativeLocalValue.capture)
    LOAD_FAST_LOAD_FAST = (auto(), NativeLocalValue.capture_many)
    LOAD_FAST_BORROW_LOAD_FAST_BORROW = (auto(), NativeLocalValue.capture_many)
    STORE_FAST = (
        auto(),
        NativeOperandStack.store,
        lambda resolver, binding: resolver._fast_local_store_resolution(binding),
    )

    BINARY_SUBSCR = (auto(), NativeSubscriptionValue.capture)
    BINARY_OP = (auto(), NativeSubscriptionValue.capture_binary)

    def __new__(cls, value: int, *arguments: object) -> Self:
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __init__(
        self,
        value: int,
        capture: Callable[
            [NativeOperandStack, NativePrimitiveOperation, dis.Instruction], None
        ] = NativeOperandStack.ignore,
        resolve_binding: (
            Callable[
                [
                    NativeBindingTransferResolverABC[NativeResolutionT],
                    NativeBindingTransfer,
                ],
                NativeResolutionT,
            ]
            | None
        ) = None,
        unpositioned_setup: bool = False,
        operand: Callable[[dis.Instruction], object] = attrgetter("argval"),
        terminates_prefix: bool = False,
        deletes_binding: bool = False,
    ) -> None:
        self._capture = capture
        self._resolve_binding = resolve_binding
        self.unpositioned_setup = unpositioned_setup
        self.operand = operand
        self.terminates_prefix = terminates_prefix
        self.deletes_binding = deletes_binding

    def require_deletion(self) -> None:
        if not self.deletes_binding:
            raise ValueError("Native binding is not a deletion")

    def capture(
        self, stack: NativeOperandStack, instruction: dis.Instruction
    ) -> ExactNativeClassPrologue | None:
        if (
            stack.expected_operation is not None
            and stack.expected_operation is not self
        ):
            raise ValueError(
                "Native prepared protocol requires its next declared operation"
            )
        stack.expected_operation = None
        self._capture(stack, self, instruction)
        if self.terminates_prefix:
            return stack.complete(instruction.offset)
        return None

    def bindings(
        self, instruction: dis.Instruction, value: NativeProducedValue | None = None
    ) -> tuple[NativeBindingTransfer, ...]:
        if self._resolve_binding is None:
            return ()
        name = self.operand(instruction)
        if type(name) is not str:
            raise ValueError("Native prologue binding has no exact string operand")
        return (
            NativeBindingTransfer(
                self,
                name,
                instruction.offset,
                instruction.arg,
                value,
                source_span=NativeInstructionSite.span_for(instruction),
            ),
        )

    def resolve_binding(
        self,
        binding: NativeBindingTransfer,
        resolver: NativeBindingTransferResolverABC[NativeResolutionT],
    ) -> NativeResolutionT:
        if self._resolve_binding is None:
            raise ValueError("Structural native operation has no binding action")
        return self._resolve_binding(resolver, binding)


@dataclass(eq=False)
class NativeCodeObservation:
    """Transient native stream with one cached initial source-body boundary."""

    code: CodeType
    _instructions: dict[int, dis.Instruction] = field(default_factory=dict, init=False)
    emissions: list[NativeCodeEmission] = field(default_factory=list, init=False)
    current: NativeCodeEmission | None = field(default=None, init=False)
    _boundary: dis.Instruction | None = field(default=None, init=False)

    @property
    def instructions(self) -> tuple[dis.Instruction, ...]:
        return tuple(self._instructions.values())

    @property
    def initial_instructions(self) -> tuple[dis.Instruction, ...]:
        return tuple(
            takewhile(
                lambda instruction: instruction is not self._boundary,
                self._instructions.values(),
            )
        )

    boundary = AliasProperty[dis.Instruction | None]("_boundary")

    def require_current_creation(
        self, instruction: dis.Instruction
    ) -> NativeCodeEmission:
        emission = self.current
        if (
            emission is None
            or emission.creation is None
            or emission.containing_code is not self.code
        ):
            raise ValueError("Native function operand has no original frame creation")
        if emission.source_span is None or None in instruction.positions:
            raise ValueError("Native function operand has incomplete source ranges")
        for event in (emission.load, emission.creation, instruction):
            if self._instructions.get(id(event)) is not event:
                raise ValueError(
                    "Native function operand requires its original instruction"
                )
        if (
            emission.load.offset >= emission.creation.offset
            or instruction.offset < emission.creation.offset
        ):
            raise ValueError("Native function operand has a nonforward creation chain")
        if instruction is not emission.creation and not any(
            event is instruction for event in emission.attachments
        ):
            raise ValueError(
                "Native function operand has no original creation attachment"
            )
        return emission

    def require_creation(self, instruction: dis.Instruction) -> NativeCodeEmission:
        """Require an original creation in the initial prefix, not a later body event."""
        if None in instruction.positions:
            raise ValueError("Native creation instruction has incomplete source ranges")
        initial = self.initial_instructions
        if not any(event is instruction for event in initial):
            raise ValueError("Native creation instruction is outside this prefix")
        candidates = tuple(
            emission
            for emission in self.emissions
            if emission.containing_code is self.code
            and emission.creation is not None
            and emission.source_span is not None
            and any(event is emission.load for event in initial)
            and any(event is emission.creation for event in initial)
            and emission.load.offset < emission.creation.offset <= instruction.offset
            and (
                instruction is emission.creation
                or any(event is instruction for event in emission.attachments)
            )
        )
        if len(candidates) != 1:
            raise ValueError("Native helper creation has no unique observed event")
        return candidates[0]

    def observe(self, instruction: dis.Instruction) -> None:
        if id(instruction) in self._instructions:
            return
        if self._boundary is None:
            header = (self.code.co_firstlineno, self.code.co_firstlineno, 0, 0)
            if (
                None not in instruction.positions
                and tuple(instruction.positions) != header
            ):
                self._boundary = instruction
        self._instructions[id(instruction)] = instruction


@dataclass(frozen=True)
class NativeCaptureSite:
    frame: NativeFrameOrigin
    instruction_offset: int

    def require_definition_operand(self, value: NativeProducedValue) -> None:
        if (
            not isinstance(value, NativeFunctionValue)
            or value.creation.instruction_offset != self.instruction_offset
        ):
            raise ValueError(
                "Native operand differs from its original function creation"
            )


@dataclass(frozen=True, eq=False)
class NativeDefinitionApplication(DataclassGraphValue, NativeCaptureSite):
    """Original zero-explicit-argument call site after raw function creation.

    The implicit argument is the original preceding creation/application site.
    Callee values, activation and invocation effects remain separate obligations.
    This is not a descriptor-value proof.
    """

    source_span: SourceByteSpan
    prelude: tuple[NativeCaptureSite, ...]
    argument: NativeCaptureSite

    def require_definition_operand(self, value: NativeProducedValue) -> None:
        if (
            not isinstance(value, NativeCallValue)
            or value.instruction_offset != self.instruction_offset
            or value.source_span != self.source_span
        ):
            raise ValueError("Native operand differs from its original application")

    def operand_in(self, store: NativeValueStore) -> NativeCallValue:
        if store.frame is not self.frame or self.argument.frame is not self.frame:
            raise ValueError("Native application operand belongs to a different frame")
        value = store.production_at(self.instruction_offset)
        self.require_definition_operand(value)
        call = cast(NativeCallValue, value)
        argument = call.require_definition_argument()
        for operand in call.inputs:
            store.require_value(operand)
        self.argument.require_definition_operand(argument)
        return call


@dataclass(frozen=True)
class NativeClassCapture(ABC):
    compilation: NativeCompilationIdentity
    source_span: SourceByteSpan

    @abstractmethod
    def resolve(
        self, resolver: NativeClassCaptureResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactNativeClassCapture(NativeClassCapture):
    """Native builder load and raw body creation, not class-body entry.

    The two instruction sites have different lookup semantics. Neither proves
    the builder's value, the body function's builtins or a prepared namespace.
    """

    body: ExactNativeFunctionExecution
    builder: NativeCaptureSite
    prologue: NativeClassPrologue

    @property
    def creation(self) -> NativeCaptureSite:
        return self.body.require_creation()

    def construction_in(self, store: NativeValueStore) -> NativeCallValue:
        """Require direct raw construction as the stored result, without decorators."""
        return self._require_construction_value(store, store.value)

    def definition_construction_in(self, store: NativeValueStore) -> NativeCallValue:
        """Join raw construction to the stored definition's original application chain.

        The returned operand is not the installed decorated result. No application
        behavior or runtime class identity follows from this compiler relation.
        """
        candidates = tuple(
            value
            for value in store.values
            if isinstance(value, NativeCallValue)
            and value.source_span == self.source_span
        )
        if len(candidates) != 1:
            raise ValueError("Native definition has no unique class construction")
        value = self._require_construction_value(store, candidates[0])
        store.applications_after(value)
        return value

    def _require_construction_value(
        self, store: NativeValueStore, value: NativeProducedValue
    ) -> NativeCallValue:
        store.require_value(value)
        if (
            store.frame is not self.builder.frame
            or self.creation.frame is not store.frame
        ):
            raise ValueError("Native class result belongs to a different creator frame")
        if (
            not isinstance(value, NativeCallValue)
            or value.source_span != self.source_span
        ):
            raise ValueError("Native class result has no original construction call")
        if (
            value.argument_slot.implicit_arguments
            or len(value.positional_arguments) < 2
        ):
            raise ValueError(
                "Native class construction requires its explicit body and name"
            )
        for operand in value.productions():
            store.require_value(operand)
        if (
            not isinstance(value.callee, NativeClassBuilderValue)
            or value.callee.instruction_offset != self.builder.instruction_offset
        ):
            raise ValueError("Native class call differs from its original builder load")
        self.creation.require_definition_operand(value.arguments[0])
        return value

    def prologue_return_operand(
        self, receipt: NativeReturn, value: NativeProducedValue
    ) -> NativeProducedValue:
        """Join two original observations at one instruction in this actual body."""
        if not receipt.frame.is_body_of(self.body):
            raise ValueError("Native return belongs to a different class body")
        receipt.require_value(value)
        if value is not receipt.value:
            raise ValueError(
                "Native tail value precedes the body and is not its returned operand"
            )
        return self.prologue.require_prior_production(value.instruction_offset)

    def resolve(
        self, resolver: NativeClassCaptureResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._exact_class_capture_resolution(self)


@dataclass(frozen=True)
class OpenNativeClassCapture(NativeClassCapture):
    reason: NativeExecutionUnavailable

    def resolve(
        self, resolver: NativeClassCaptureResolverABC[NativeResolutionT]
    ) -> NativeResolutionT:
        return resolver._open_class_capture_resolution(self)


@dataclass(eq=False)
class NativeInstructionSite:
    """One transient actual instruction in its actual containing code."""

    containing_code: CodeType
    instruction: dis.Instruction

    @property
    def source_span(self) -> SourceByteSpan | None:
        return self.span_for(self.instruction)

    @staticmethod
    def span_for(instruction: dis.Instruction) -> SourceByteSpan | None:
        line, end_line, column, end_column = instruction.positions
        if None in (line, end_line, column, end_column):
            return None
        return SourceByteSpan(line - 1, end_line - 1, column, end_column)


@dataclass(frozen=True, eq=False)
class NativeReturn(DataclassGraphValue, NativeValueInventoryABC):
    """Conditional straight-line continuation in one original native frame.

    Shared by stores in the same uninterrupted suffix. Source activation,
    namespace lookup, transfer effects and construction remain separate proofs.
    """

    frame: NativeFrameOrigin
    instruction_offset: int
    value: NativeProducedValue
    bindings: tuple[NativeBindingTransfer, ...]
    stores: tuple[NativeBindingTransfer, ...]

    values: tuple[NativeProducedValue, ...] = field(default=(), kw_only=True)

    def require_from_entry(self) -> NativeReturn:
        """Require native continuity from entry, independently of source effects."""
        raise ValueError("Native return suffix does not prove continuity from entry")

    def effects_for(
        self, span: SourceByteSpan, declaration: type[NativeEffectT]
    ) -> tuple[NativeEffectT, ...]:
        return tuple(
            value
            for value in self.values
            if isinstance(value, declaration) and value.source_span == span
        )

    def effect_for(
        self, span: SourceByteSpan, declaration: type[NativeEffectT]
    ) -> NativeEffectT:
        candidates = self.effects_for(span, declaration)
        if len(candidates) != 1:
            raise ValueError("Native stack effect has no unique original production")
        (effect,) = candidates
        self.require_value(effect)
        if (
            len(effect.inputs) != effect.operand_count
            or effect.instruction_offset >= self.instruction_offset
        ):
            raise ValueError(
                "Native stack effect has no original operands before return"
            )
        for value in effect.productions():
            self.require_value(value)
        return effect

    @cached_property
    def bindings_by_source(
        self,
    ) -> dict[tuple[SourceByteSpan | None, str], NativeBindingTransfer]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.stores, lambda binding: (binding.source_span, binding.name)
        )

    def binding_for(self, span: SourceByteSpan, name: str) -> NativeBindingTransfer:
        key = (span, name)
        if key not in self.bindings_by_source:
            raise ValueError(
                "Native continuation has no unique original source binding"
            )
        binding = self.bindings_by_source[key]
        if not self.continues(binding):
            raise ValueError("Native source binding has an ambiguous instruction")
        return binding

    @cached_property
    def _stores_by_offset(self) -> dict[int, NativeBindingTransfer]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.stores, lambda binding: binding.instruction_offset
        )

    def continues(self, binding: NativeBindingTransfer) -> bool:
        return self._stores_by_offset.get(binding.instruction_offset) is binding

    def require_store(self, store: NativeValueStore) -> None:
        if (
            store.continuation is not self
            or store.frame is not self.frame
            or not self.continues(store.binding)
        ):
            raise ValueError(
                "Native return does not continue this store in its original frame"
            )

    def after(self, store: NativeValueStore) -> tuple[NativeBindingTransfer, ...]:
        self.require_store(store)
        return self.after_binding(store.binding)

    def after_binding(
        self, installed: NativeBindingTransfer
    ) -> tuple[NativeBindingTransfer, ...]:
        if not self.continues(installed):
            raise ValueError("Native tail requires an original installed binding")
        return tuple(
            binding
            for binding in self.bindings
            if binding.instruction_offset > installed.instruction_offset
        )


@dataclass(frozen=True, eq=False)
class NativeValueStore(DataclassGraphValue, NativeValueInventoryABC):
    """An actual operand graph and result store in its original frame."""

    frame: NativeFrameOrigin
    production_span: SourceByteSpan
    binding: NativeBindingTransfer

    continuation: NativeReturn | None = field(default=None, kw_only=True)

    @property
    def values(self) -> tuple[NativeProducedValue, ...]:
        return self.value.productions()

    @property
    def source_span(self) -> SourceByteSpan:
        span = self.binding.source_span
        if span is None:
            raise ValueError("Native store has no original target span")
        return span

    def require_return(self) -> NativeReturn:
        if self.continuation is None:
            raise ValueError("Native store has no proved straight-line return")
        self.continuation.require_store(self)
        return self.continuation

    def applications_after(
        self, creation: NativeProducedValue
    ) -> tuple[NativeCallValue, ...]:
        """Derive the inner-to-outer application path from original stored operands.

        A reachable value used as a callee or explicit argument is not a definition
        predecessor. Native production ordering also rejects cycles without an
        independently retained traversal state or a second application registry.
        """
        self.require_value(creation)
        current = self.value
        applications = []
        while current is not creation:
            self.require_value(current)
            if not isinstance(current, NativeCallValue):
                raise ValueError("Native definition has no application path to storage")
            argument = current.require_definition_argument()
            for operand in current.inputs:
                self.require_value(operand)
                if operand.instruction_offset >= current.instruction_offset:
                    raise ValueError("Native application must follow its operands")
            applications.append(current)
            current = argument
        return tuple(reversed(applications))

    @property
    def value(self) -> NativeProducedValue:
        value = self.binding.value
        if value is None:
            raise ValueError("Native value store has no production")
        return value


class NativeConstantStore(NativeValueStore):
    """Entry-only, same-span text storage used by compiler documentation."""

    @property
    def value(self) -> NativeConstantValue:
        value = super().value.require_scalar_store_value()
        value.require_native_text()
        return value


class NativeEntryReturn(NativeReturn):
    """The original complete primitive walk reaches this return from frame entry.

    Argument binding, invocation, lookup, effects and cleanup remain separate
    obligations. No target code was executed to obtain this compiler receipt.
    """

    def require_from_entry(self) -> NativeReturn:
        return self


class NativeOperandObservation(ABC):
    """The observation's original code supplies its transient operand interpreter."""

    code: CodeType | None

    @cached_property
    def operands(self) -> NativeOperandStack:
        return NativeOperandStack(code=self.code)


@dataclass(eq=False)
class NativeValueStoreWindow(NativeOperandObservation):
    """Observe one complete operand graph and its consuming native store.

    Every production must feed the stored root and all operands must be consumed.
    Unknown operations, jump entries and backwards instructions close the window.
    Source value identity and execution effects remain separate obligations.
    """

    code: CodeType
    operations: dict[int, NativePrimitiveOperation]
    load: NativeInstructionSite | None = None
    store: NativeInstructionSite | None = None
    closed: bool = False

    receipt_type: ClassVar[type[NativeValueStore]] = NativeValueStore
    creation_context: NativeCodeObservation | None = field(default=None, kw_only=True)

    def capture_instruction(self, instruction: dis.Instruction) -> None:
        operation = self.operations.get(instruction.opcode)
        if operation is not None and operation.terminates_prefix:
            raise ValueError("Native operand window reached its transfer boundary")
        context = self.creation_context
        if context is not None and context.code is not self.code:
            raise ValueError("Native operand window has no original creation context")
        self.operands.capture_instruction(instruction, operation, context)

    def require_production(self, value: NativeProducedValue) -> None:
        """Operand observation retains provenance; source value and effects stay separate."""
        pass

    def require_store(self, store: NativeInstructionSite) -> None:
        if (
            self.load is None
            or self.load.source_span is None
            or store.source_span is None
        ):
            raise ValueError(
                "Native store requires complete production and target spans"
            )

    def observe(self, instruction: dis.Instruction) -> None:
        if self.closed:
            return
        if instruction.is_jump_target or (
            self.load is not None and instruction.offset <= self.load.instruction.offset
        ):
            self.closed = True
            return
        try:
            self.capture_instruction(instruction)
        except (ValueError, TypeError):
            self.closed = True
            return
        if self.operands.bindings:
            self.closed = True
            if len(self.operands.bindings) != 1 or self.operands.stack:
                return
            value = self.operands.bindings[0].value
            if value is None:
                return
            store = NativeInstructionSite(self.code, instruction)
            try:
                self.require_store(store)
                reachable = {id(production) for production in value.productions()}
            except ValueError:
                return
            if reachable != {id(production) for production in self.operands.values}:
                return
            self.store = store
        elif (
            self.operands.values
            and self.operands.values[-1].instruction_offset == instruction.offset
        ):
            try:
                self.require_production(self.operands.values[-1])
            except ValueError:
                self.closed = True
                return
            self.load = NativeInstructionSite(self.code, instruction)

    def receipt(
        self, frame: NativeFrameOrigin, continuation: NativeReturn | None = None
    ) -> NativeValueStore | None:
        if self.store is None:
            return None
        assert self.load is not None
        return self.receipt_type(
            frame,
            cast(SourceByteSpan, self.load.source_span),
            self.operands.bindings[0],
            continuation=(
                continuation
                if continuation is not None
                and continuation.continues(self.operands.bindings[0])
                else None
            ),
        )


class NativeEntryValueWindow(NativeValueStoreWindow):
    """Retain the entry-only text and same-span documentation contract."""

    receipt_type = NativeConstantStore

    def require_production(self, value: NativeProducedValue) -> None:
        if self.load is not None:
            raise ValueError("Native entry documentation requires a single production")
        value.require_scalar_store_value().require_native_text()

    def require_store(self, store: NativeInstructionSite) -> None:
        super().require_store(store)
        assert self.load is not None
        if self.load.source_span != store.source_span:
            raise ValueError("Native entry documentation requires one source span")


@dataclass
class NativeContinuationWindow(NativeOperandObservation):
    """One uninterrupted operand walk owns original stores and residual operands."""

    stores: dict[int, NativeBindingTransfer] = field(default_factory=dict)
    code: CodeType | None = field(default=None, kw_only=True)
    creation_context: NativeCodeObservation | None = field(default=None, kw_only=True)
    last_instruction_offset: int = field(default=-1, init=False)

    def record_store(self, binding: NativeBindingTransfer) -> NativeBindingTransfer:
        original = self.stores.setdefault(binding.instruction_offset, binding)
        original.require_observation(binding)
        return original

    def observe(
        self, instruction: dis.Instruction, operation: NativePrimitiveOperation | None
    ) -> None:
        if (
            instruction.is_jump_target
            or self.operands.returned is not None
            or instruction.offset <= self.last_instruction_offset
        ):
            raise ValueError(
                "Native continuation is not an uninterrupted primitive suffix"
            )
        self.last_instruction_offset = instruction.offset
        before = len(self.operands.bindings)
        self.operands.capture_instruction(instruction, operation, self.creation_context)
        for binding in self.operands.bindings[before:]:
            self.record_store(binding)

    def receipt(
        self,
        frame: NativeFrameOrigin,
        receipt_type: type[NativeReturn] = NativeReturn,
    ) -> NativeReturn | None:
        value = self.operands.returned
        offset = self.operands.return_offset
        if value is None or offset is None:
            return None
        return receipt_type(
            frame,
            offset,
            value,
            tuple(self.operands.bindings),
            tuple(self.stores.values()),
            values=tuple(self.operands.values),
        )


@dataclass
class NativeStoreStream:
    """One continuation observer consuming original stores from their owners."""

    code: CodeType
    operations: dict[int, NativePrimitiveOperation]
    continuation: NativeContinuationWindow | None = field(default=None, init=False)
    creation_context: NativeCodeObservation | None = field(default=None, kw_only=True)

    def new_continuation(self) -> NativeContinuationWindow:
        context = self.creation_context
        if context is not None and context.code is not self.code:
            raise ValueError("Native stream has no original creation context")
        return NativeContinuationWindow(code=self.code, creation_context=context)

    def record_store(self, binding: NativeBindingTransfer) -> NativeBindingTransfer:
        if self.continuation is None:
            self.continuation = self.new_continuation()
        return self.continuation.record_store(binding)

    @property
    def receipt_type(self) -> type[NativeReturn]:
        return NativeReturn

    def return_receipt(self, frame: NativeFrameOrigin) -> NativeReturn | None:
        return (
            None
            if self.continuation is None
            else self.continuation.receipt(frame, self.receipt_type)
        )

    def observe(self, instruction: dis.Instruction) -> None:
        if self.continuation is None:
            self.continuation = self.new_continuation()
        try:
            self.continuation.observe(
                instruction, self.operations.get(instruction.opcode)
            )
        except (ValueError, TypeError):
            self.continuation = None


@dataclass
class NativeValueStoreStream(NativeStoreStream):
    """Derive value/store receipts from the same segments that own continuations."""

    segments: list[NativeContinuationWindow] = field(default_factory=list)

    @property
    def receipt_type(self) -> type[NativeReturn]:
        return (
            NativeEntryReturn
            if len(self.segments) == 1 and self.continuation is self.segments[0]
            else super().receipt_type
        )

    def new_continuation(self) -> NativeContinuationWindow:
        segment = super().new_continuation()
        self.segments.append(segment)
        return segment

    def receipts(
        self, frame: NativeFrameOrigin, continuation: NativeReturn | None
    ) -> tuple[NativeValueStore, ...]:
        return tuple(
            NativeValueStore(
                frame,
                binding.value.source_span,
                binding,
                continuation=(
                    continuation
                    if continuation is not None and continuation.continues(binding)
                    else None
                ),
            )
            for segment in self.segments
            for binding in segment.stores.values()
            if binding.value is not None
            and binding.value.source_span is not None
            and binding.source_span is not None
        )


@dataclass(frozen=True)
class NativeScopeReceipts:
    """One publication of an observed scope's original frame, stores and return."""

    frame: NativeFrameOrigin
    constant_stores: tuple[NativeConstantStore, ...]
    value_stores: tuple[NativeValueStore, ...]
    continuation: NativeReturn | None


@dataclass(eq=False)
class NativeScopeObservation(NativeCodeObservation):
    """One original walk owns creation, entry and continuation observations."""

    operations: dict[int, NativePrimitiveOperation]

    @cached_property
    def entry_window(self) -> NativeEntryValueWindow:
        return NativeEntryValueWindow(self.code, self.operations)

    @cached_property
    def value_stream(self) -> NativeValueStoreStream:
        return NativeValueStoreStream(self.code, self.operations, creation_context=self)

    def observe_values(
        self,
        instruction: dis.Instruction,
        previous: NativeCodeEmission | None,
        *,
        after_prologue: bool,
    ) -> None:
        self.value_stream.observe(instruction)
        if after_prologue:
            self.entry_window.observe(instruction)
        if previous is not None and previous.installation is not None:
            previous.installation = self.value_stream.record_store(
                previous.installation
            )

    def publish(self, frame: NativeFrameOrigin) -> NativeScopeReceipts:
        continuation = self.value_stream.return_receipt(frame)
        constant = self.entry_window.receipt(frame)
        return NativeScopeReceipts(
            frame,
            () if constant is None else (constant,),
            self.value_stream.receipts(frame, continuation),
            continuation,
        )


class NativeScopeInventoryABC(ABC):
    """Flat query views derive from scope publications, never separate facts."""

    @property
    @abstractmethod
    def scopes(self) -> tuple[NativeScopeReceipts, ...]:
        raise NotImplementedError

    @property
    def constant_stores(self) -> tuple[NativeConstantStore, ...]:
        return tuple(store for scope in self.scopes for store in scope.constant_stores)

    @property
    def value_stores(self) -> tuple[NativeValueStore, ...]:
        return tuple(store for scope in self.scopes for store in scope.value_stores)

    @property
    def returns(self) -> tuple[NativeReturn, ...]:
        return tuple(
            scope.continuation
            for scope in self.scopes
            if scope.continuation is not None
        )


@dataclass(eq=False)
class NativeCreationInventory(NativeScopeInventoryABC):
    """Transient results of the one complete emitted-code instruction walk."""

    root_code: CodeType
    emissions: tuple[NativeCodeEmission, ...]
    builder_loads: tuple[NativeInstructionSite, ...]

    prefixes: dict[int, NativeCodeObservation]

    observations: tuple[NativeScopeObservation, ...] = ()

    frame_origins: dict[int, NativeFrameOrigin] = field(
        default_factory=dict, init=False
    )

    scopes: tuple[NativeScopeReceipts, ...] = field(default=(), init=False)

    def bind_frame_origins(
        self,
        compilation: NativeCompilationIdentity,
        selected: tuple[NativeCodeEmission, ...],
    ) -> None:
        """One code-identity authority joins both creator and executing-frame receipts."""
        owners = {id(emission): emission for emission in selected}
        by_code = UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            owners.values(),
            lambda emission: id(emission.code),
        )
        self.frame_origins = {id(self.root_code): ModuleNativeFrameOrigin(compilation)}

        def origin_for(identity: int) -> NativeFrameOrigin:
            if identity not in self.frame_origins:
                owner = by_code.unambiguous_declarations_by_handle.get(identity)
                self.frame_origins[identity] = (
                    OpenNativeFrameOrigin(
                        compilation,
                        (
                            NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
                            if identity in by_code.ambiguous_handles
                            else NativeExecutionUnavailable.UNJOINED_FRAME_ORIGIN
                        ),
                    )
                    if owner is None
                    else SourceNativeFrameOrigin(
                        cast(ExactNativeFunctionExecution, owner.receipt)
                    )
                )
            return self.frame_origins[identity]

        for emission in self.emissions:
            emission.binding = NativeEmissionBinding(
                emission,
                origin_for(id(emission.containing_code)),
            )
        self.scopes = tuple(
            observation.publish(origin_for(id(observation.code)))
            for observation in self.observations
        )

    def class_captures(
        self,
        compilation: NativeCompilationIdentity,
        selected: tuple[NativeCodeEmission, ...],
        backend: NativeCreationBackend,
    ) -> dict[SourceByteSpan, NativeClassCapture]:
        builders: dict[SourceByteSpan, list[NativeInstructionSite]] = {}
        emissions: dict[tuple[SourceByteSpan, int], list[NativeCodeEmission]] = {}
        for builder in self.builder_loads:
            if builder.source_span is not None:
                builders.setdefault(builder.source_span, []).append(builder)
        for emission in self.emissions:
            if emission.source_span is not None:
                key = (emission.source_span, id(emission.containing_code))
                emissions.setdefault(key, []).append(emission)
        captures: dict[
            SourceByteSpan, tuple[NativeInstructionSite, NativeCodeEmission]
        ] = {}
        results: dict[SourceByteSpan, NativeClassCapture] = {}
        for span, sites in builders.items():
            if len(sites) != 1:
                results[span] = OpenNativeClassCapture(
                    compilation, span, NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
                )
                continue
            builder = sites[0]
            candidates = emissions.get((span, id(builder.containing_code)), ())
            if len(candidates) != 1:
                results[span] = OpenNativeClassCapture(
                    compilation, span, NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
                )
                continue
            body = candidates[0]
            if not backend.proves_class_capture(builder, body):
                results[span] = OpenNativeClassCapture(
                    compilation, span, NativeExecutionUnavailable.NO_OBSERVED_CREATION
                )
                continue
            captures[span] = (builder, body)
        self.bind_frame_origins(
            compilation, (*selected, *(body for _, body in captures.values()))
        )
        for span, (builder, body) in captures.items():
            origin = body.frame_origin
            results[span] = ExactNativeClassCapture(
                compilation,
                span,
                cast(ExactNativeFunctionExecution, body.receipt),
                NativeCaptureSite(origin, builder.instruction.offset),
                backend.class_prologue(self.prefixes[id(body.code)]),
            )
        return results


@dataclass(frozen=True)
class _NativeCompilationOutcome(ABC):
    compilation: NativeCompilationIdentity

    @property
    def module_annotation_setup(self) -> NativeValueStore | None:
        raise ValueError("Native module entry work remains unproved")

    def return_after_binding(self, span: SourceByteSpan, name: str) -> NativeReturn:
        raise ValueError("Native source binding has no proved return continuation")

    def return_after_effect(
        self, span: SourceByteSpan, declaration: type[NativeStackEffectABC]
    ) -> NativeReturn:
        raise ValueError("Native stack effect has no proved return continuation")

    def prologue_return_operand(
        self,
        capture: ExactNativeClassCapture,
        receipt: NativeReturn,
        value: NativeProducedValue,
    ) -> NativeProducedValue:
        if (
            self.class_capture_for(capture.source_span) is not capture
            or self.return_from(capture.body) is not receipt
        ):
            raise ValueError(
                "Native prologue join requires original compilation receipts"
            )
        return capture.prologue_return_operand(receipt, value)

    def value_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        raise ValueError("Native value store remains unproved")

    def return_from(self, execution: NativeFunctionExecution) -> NativeReturn:
        raise ValueError("Native body has no proved return continuation")

    def require_function(self, execution: NativeFunctionExecution) -> None:
        if (
            execution.compilation is not self.compilation
            or self.execution_for(execution.source_span) is not execution
        ):
            raise ValueError(
                "Native operation requires its canonical compilation receipt"
            )

    def return_after(self, execution: NativeFunctionExecution) -> NativeReturn:
        raise ValueError("Native function store has no proved return continuation")

    def scalar_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        raise ValueError("Native scalar store remains unproved")

    def constant_store_for(
        self, source_span: SourceByteSpan, name: str
    ) -> NativeConstantStore:
        raise ValueError("Native constant store remains unproved")

    @abstractmethod
    def class_capture_for(self, source_span: SourceByteSpan) -> NativeClassCapture:
        raise NotImplementedError

    @abstractmethod
    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        raise NotImplementedError


class _RejectedNativeCompilation(_NativeCompilationOutcome):

    def class_capture_for(self, source_span: SourceByteSpan) -> NativeClassCapture:
        return OpenNativeClassCapture(
            self.compilation,
            source_span,
            NativeExecutionUnavailable.COMPILATION_REJECTED,
        )

    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        return OpenNativeFunctionExecution(
            self.compilation,
            source_span,
            NativeExecutionUnavailable.COMPILATION_REJECTED,
        )


@dataclass(frozen=True)
class _NativeExecutionIndex(NativeScopeInventoryABC, _NativeCompilationOutcome):
    declarations: IdentityHandleMultiplicityProjection[
        SourceByteSpan, ExactNativeFunctionExecution
    ]
    has_incomplete_ranges: bool

    class_captures: dict[SourceByteSpan, NativeClassCapture]

    class_capture_fallback: NativeExecutionUnavailable

    scopes: tuple[NativeScopeReceipts, ...] = ()

    @cached_property
    def module_annotation_setup(self) -> NativeValueStore | None:
        """Original root SETUP_ANNOTATIONS receipt, independent of later stores."""
        if self.has_incomplete_ranges:
            raise ValueError("Native module entry has incomplete source ranges")
        candidates = tuple(
            store
            for store in self.value_stores
            if store.frame == ModuleNativeFrameOrigin(self.compilation)
            and isinstance(store.value, NativeAnnotationNamespaceValue)
        )
        if len(candidates) > 1:
            raise ValueError("Native module annotation setup is ambiguous")
        return candidates[0] if candidates else None

    def return_after_binding(self, span: SourceByteSpan, name: str) -> NativeReturn:
        key = (span, name)
        candidates = tuple(
            receipt
            for receipt in self.returns
            for binding in receipt.stores
            if (binding.source_span, binding.name) == key
        )
        if self.has_incomplete_ranges or len(candidates) != 1:
            raise ValueError("Native source binding has no unique return continuation")
        receipt = candidates[0]
        receipt.binding_for(span, name)
        return receipt

    def scalar_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        receipt = self.value_store_for(production_span, target_span, name)
        receipt.value.require_scalar_store_value()
        return receipt

    def return_after_effect(
        self, span: SourceByteSpan, declaration: type[NativeStackEffectABC]
    ) -> NativeReturn:
        candidates = tuple(
            receipt
            for receipt in self.returns
            if receipt.effects_for(span, declaration)
        )
        if self.has_incomplete_ranges or len(candidates) != 1:
            raise ValueError("Native stack effect has no unique return continuation")
        (receipt,) = candidates
        receipt.effect_for(span, declaration)
        return receipt

    def return_from(self, execution: NativeFunctionExecution) -> NativeReturn:
        self.require_function(execution)
        candidates = tuple(
            receipt for receipt in self.returns if receipt.frame.is_body_of(execution)
        )
        if len(candidates) != 1:
            raise ValueError("Native body has no unique return continuation")
        return candidates[0]

    def return_after(self, execution: NativeFunctionExecution) -> NativeReturn:
        self.require_function(execution)
        frame = execution.require_creation().frame
        binding = execution.require_installation()
        candidates = tuple(
            receipt
            for receipt in self.returns
            if receipt.frame is frame and receipt.continues(binding)
        )
        if len(candidates) != 1:
            raise ValueError("Native function store has no unique return continuation")
        return candidates[0]

    @cached_property
    def _value_stores_by_key(
        self,
    ) -> dict[tuple[SourceByteSpan, SourceByteSpan, str], NativeValueStore]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.value_stores,
            lambda store: (
                store.production_span,
                store.source_span,
                store.binding.name,
            ),
        )

    def value_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        key = (production_span, target_span, name)
        if self.has_incomplete_ranges or key not in self._value_stores_by_key:
            raise ValueError("Native value store has no unique original receipt")
        return self._value_stores_by_key[key]

    @cached_property
    def _constant_stores_by_key(
        self,
    ) -> dict[tuple[SourceByteSpan, str], NativeConstantStore]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.constant_stores,
            lambda store: (store.source_span, store.binding.name),
        )

    def constant_store_for(
        self,
        source_span: SourceByteSpan,
        name: str,
    ) -> NativeConstantStore:
        key = (source_span, name)
        if self.has_incomplete_ranges or key not in self._constant_stores_by_key:
            raise ValueError("Native constant store has no unique original receipt")
        return self._constant_stores_by_key[key]

    def class_capture_for(self, source_span: SourceByteSpan) -> NativeClassCapture:
        if self.has_incomplete_ranges:
            return OpenNativeClassCapture(
                self.compilation,
                source_span,
                NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES,
            )
        if source_span in self.class_captures:
            return self.class_captures[source_span]
        return OpenNativeClassCapture(
            self.compilation, source_span, self.class_capture_fallback
        )

    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        reason = self.unavailability_for(source_span)
        if reason is not None:
            return OpenNativeFunctionExecution(self.compilation, source_span, reason)
        return self.declarations.unambiguous_declarations_by_handle[source_span]

    def unavailability_for(
        self, source_span: SourceByteSpan
    ) -> NativeExecutionUnavailable | None:
        if self.has_incomplete_ranges:
            return NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES
        if source_span in self.declarations.ambiguous_handles:
            return NativeExecutionUnavailable.AMBIGUOUS_SOURCE_SPAN
        if source_span not in self.declarations.unambiguous_declarations_by_handle:
            return NativeExecutionUnavailable.NO_EMITTED_CODE
        return None


@dataclass(eq=False)
class NativeCodeEmission:
    """One actual load site, with its observed native creation/attachment events.

    Identity is this emission, never CodeType equality, flags or source span.
    All fields are transient. A frame binding owns its compact receipt lifetime.
    """

    compilation: NativeCompilationIdentity
    load: dis.Instruction
    containing_code: CodeType
    creation: dis.Instruction | None = None
    binding: NativeEmissionBinding = field(init=False, repr=False)
    attachments: list[dis.Instruction] = field(default_factory=list)
    applications: list[NativeDefinitionApplicationObservation] = field(
        default_factory=list
    )
    application_prelude: list[dis.Instruction] = field(default_factory=list)

    receipt = AliasProperty[ExactNativeFunctionExecution | None]("binding.receipt")

    frame_origin = AliasProperty[NativeFrameOrigin]("binding.frame")

    installation: NativeBindingTransfer | None = None

    @property
    def code(self) -> CodeType:
        return cast(CodeType, self.load.argval)

    @cached_property
    def source_span(self) -> SourceByteSpan | None:
        return NativeInstructionSite(self.containing_code, self.load).source_span


@dataclass(frozen=True)
class NativeDefinitionApplicationObservation:
    """Transient original instructions joined to the emission's selected frame."""

    call: dis.Instruction
    prelude: tuple[dis.Instruction, ...]

    def receipt(self, argument: NativeCaptureSite) -> NativeDefinitionApplication:
        span = NativeInstructionSite.span_for(self.call)
        if span is None:
            raise ValueError("Native application requires its original source range")
        offsets = (
            argument.instruction_offset,
            *(step.offset for step in self.prelude),
            self.call.offset,
        )
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise ValueError(
                "Native application must follow its original implicit argument"
            )
        frame = argument.frame
        return NativeDefinitionApplication(
            frame,
            self.call.offset,
            span,
            tuple(NativeCaptureSite(frame, step.offset) for step in self.prelude),
            argument,
        )


@dataclass(frozen=True, eq=False)
class NativeEmissionBinding:
    """One emission/frame join owns its canonical compact execution receipt.

    A different selection creates a new binding. Previously returned receipts
    retain their original proof; no cached receipt crosses binding lifetimes.
    """

    emission: NativeCodeEmission
    frame: NativeFrameOrigin

    @cached_property
    def receipt(self) -> ExactNativeFunctionExecution | None:
        source = self.emission
        span = source.source_span
        if span is None:
            return None
        arguments = (source.compilation, span, source.code.co_flags)
        if (
            source.creation is None
            or NativeInstructionSite(
                source.containing_code, source.creation
            ).source_span
            is None
        ):
            return ExactNativeFunctionExecution(*arguments)
        creation = NativeCaptureSite(self.frame, source.creation.offset)
        if source.installation is None:
            return CreatedNativeFunctionExecution(*arguments, creation)
        if source.applications:
            applications: list[NativeDefinitionApplication] = []
            argument = creation
            for observation in source.applications:
                argument = observation.receipt(argument)
                applications.append(argument)
            return AppliedNativeFunctionExecution(
                *arguments,
                creation,
                tuple(applications),
                source.installation,
            )
        return InstalledNativeFunctionExecution(
            *arguments, creation, source.installation
        )


class NativeCreationOperation(ABC, metaclass=AutoRegisterMeta):
    """Native creation-chain transfers registered from their declarations."""

    __registry__: ClassVar[dict[str, type[NativeCreationOperation]]] = {}
    __registry_key__ = "native_name"
    __skip_if_no_key__ = True
    native_name: ClassVar[str]

    @classmethod
    def capture_prologue(
        cls,
        stack: NativeOperandStack,
        prefix: NativeCodeObservation,
        instruction: dis.Instruction,
    ) -> None:
        raise ValueError("Native creation has no prologue value proof")

    @classmethod
    def capture_operands(
        cls,
        stack: NativeOperandStack,
        emission: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> None:
        raise ValueError("Native creation operation has no function operand proof")

    @classmethod
    @abstractmethod
    def advance(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        raise NotImplementedError


class CreatedNativeFunctionOperation(NativeCreationOperation):
    """Transfers that require an existing original function creation."""

    @classmethod
    def advance(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        if current is None or current.creation is None:
            return None
        return cls._advance_created(backend, current, instruction)

    @classmethod
    @abstractmethod
    def _advance_created(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> NativeCodeEmission | None:
        raise NotImplementedError


class LoadNativeCode(NativeCreationOperation):
    native_name = "LOAD_CONST"

    @classmethod
    def advance(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        return emission


class NativeFunctionCreationOperation(NativeCreationOperation):
    """Original function creation/attachment produces one exact function value."""

    @classmethod
    def capture_prologue(
        cls,
        stack: NativeOperandStack,
        prefix: NativeCodeObservation,
        instruction: dis.Instruction,
    ) -> None:
        emission = prefix.require_creation(instruction)
        cls.capture_operands(stack, emission, instruction)


class MakeNativeFunction(NativeFunctionCreationOperation):
    native_name = "MAKE_FUNCTION"

    @classmethod
    def capture_operands(
        cls,
        stack: NativeOperandStack,
        emission: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> None:
        inputs = stack.pop(1 - dis.stack_effect(instruction.opcode, instruction.arg))
        if not inputs or inputs[-1].instruction_offset != emission.load.offset:
            raise ValueError("Native function creation has no original code operand")
        code = inputs[-1]
        if (
            not isinstance(code, NativeTypedValue)
            or code.native_type is not CodeType
            or code.source_span != emission.source_span
        ):
            raise ValueError("Native function creation has a different code operand")
        stack.stack.append(
            stack.emit(NativeFunctionValue(instruction.offset, inputs), instruction)
        )

    @classmethod
    def advance(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        if current is None or current.creation is not None:
            return None
        current.creation = instruction
        return current


class AttachNativeFunctionAttribute(
    CreatedNativeFunctionOperation, NativeFunctionCreationOperation
):
    native_name = "SET_FUNCTION_ATTRIBUTE"

    @classmethod
    def capture_operands(
        cls,
        stack: NativeOperandStack,
        emission: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> None:
        inputs = stack.pop(2)
        function = inputs[-1]
        if (
            not isinstance(function, NativeFunctionValue)
            or function.creation.instruction_offset != emission.creation.offset
        ):
            raise ValueError(
                "Native function attachment has a different original function"
            )
        stack.stack.append(
            stack.emit(
                NativeFunctionAttributeValue(
                    instruction.offset, inputs, instruction.arg
                ),
                instruction,
            )
        )

    @classmethod
    def _advance_created(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> NativeCodeEmission | None:
        if (
            current.applications
            or current.application_prelude
            or instruction.arg not in backend.attribute_flags
        ):
            return None
        current.attachments.append(instruction)
        return current


class PrepareNativeDefinitionApplication(CreatedNativeFunctionOperation):
    native_name = "PRECALL"

    @classmethod
    def _advance_created(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> NativeCodeEmission | None:
        current.application_prelude.append(instruction)
        return current


class ApplyNativeDefinition(CreatedNativeFunctionOperation):
    native_name = "CALL"

    @classmethod
    def _advance_created(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> NativeCodeEmission | None:
        try:
            backend.require_definition_application(
                current.application_prelude, instruction
            )
        except ValueError:
            return None
        current.applications.append(
            NativeDefinitionApplicationObservation(
                instruction, tuple(current.application_prelude)
            )
        )
        current.application_prelude.clear()
        return current


class InstallNativeFunction(CreatedNativeFunctionOperation):
    """End a contiguous definition-result chain at its declared storage operation."""

    operation: ClassVar[NativePrimitiveOperation] = NativePrimitiveOperation.STORE_NAME
    native_name = operation.name

    @classmethod
    def _advance_created(
        cls,
        backend: ContiguousNativeCreationBackend,
        current: NativeCodeEmission,
        instruction: dis.Instruction,
    ) -> NativeCodeEmission | None:
        if current.source_span is None or current.application_prelude:
            return None
        if (
            NativeInstructionSite(current.containing_code, instruction).source_span
            != current.source_span
        ):
            return None
        try:
            (current.installation,) = cls.operation.bindings(instruction)
        except ValueError:
            return None
        return None


class InstallGlobalNativeFunction(InstallNativeFunction):
    operation = NativePrimitiveOperation.STORE_GLOBAL
    native_name = operation.name


class InstallFastNativeFunction(InstallNativeFunction):
    operation = NativePrimitiveOperation.STORE_FAST
    native_name = operation.name


class InstallCellNativeFunction(InstallNativeFunction):
    operation = NativePrimitiveOperation.STORE_DEREF
    native_name = operation.name


class NativeCreationBackend(ABC, metaclass=AutoRegisterMeta):
    """Compiler-owned role evidence with a shared complete-emission inventory."""

    __registry__: ClassVar[
        dict[tuple[str, tuple[int, int]], type[NativeCreationBackend]]
    ] = {}
    __registry_key__ = "compiler_identity"
    __skip_if_no_key__ = True
    compiler_identity: ClassVar[tuple[str, tuple[int, int]]]

    @staticmethod
    def code_contents(code: CodeType) -> tuple[bytes, tuple[int, ...]]:
        """Native contents plus constant alias relations, without target callbacks.

        Format 2 removes serializer interning/reference-layout differences. The
        original constant graph separately retains repeated-object relationships;
        equal serialized values alone would erase observable constant sharing.
        This projection does not identify any object with an external reference.
        """
        payload = marshal.dumps(code, 2)
        handles: dict[int, int] = {}
        relations: list[int] = []

        def visit(value: object) -> None:
            identity = id(value)
            if identity in handles:
                relations.append(handles[identity])
                return
            handles[identity] = len(handles)
            relations.append(handles[identity])
            if type(value) is CodeType:
                visit(value.co_consts)
            elif type(value) in (tuple, frozenset):
                for item in value:
                    visit(item)

        visit(code)
        return payload, tuple(relations)

    def require_static_type_mro(self, value_type: type) -> tuple[type, ...]:
        """Authenticate each original native MRO owner once for lookup consumers."""
        self.require_static_type_release(value_type)
        owners = NativeClassMroDeclaration.native_mro(value_type)
        for owner in owners[1:]:
            self.require_static_type_release(owner)
        return owners

    def require_invocation(
        self, prelude: list[dis.Instruction], call: dis.Instruction
    ) -> None:
        raise ValueError("Native call protocol remains unproved")

    @property
    def call_operand_order(self) -> NativeCallOperandOrder:
        raise ValueError("Native call operand layout remains unproved")

    @property
    def binary_subscription_argument(self) -> int:
        raise ValueError("Native binary subscription encoding remains unproved")

    def require_fresh_class_namespace(
        self, metaclass: NativeClassMroDeclaration
    ) -> None:
        """Conditional fresh exact-dict preparation, not source activation or construction."""
        raise ValueError("Native fresh class namespace remains unproved")

    def require_fresh_function_namespace(
        self, compilation: NativeCompilationIdentity
    ) -> None:
        """Conditional empty custom storage at raw native function creation only."""
        raise ValueError("Native fresh function namespace remains unproved")

    def require_nonabstract_member_type(self, value_type: type) -> None:
        """Require inert absence of ABCMeta's marker in the native type lookup.

        Instance storage is a separate obligation; exact type alone does not
        establish the absence of custom function attributes.
        """
        raise ValueError("Native abstract-member lookup remains unproved")

    def require_nonabstract_member(self, value_type: type) -> None:
        """Require marker absence without relying on unproved instance storage."""
        raise ValueError("Native abstract-member instance storage remains unproved")

    def require_classvar_binding(self, argument: InertNativeArgumentWitness) -> None:
        """Require value-dependent native binding, independently of result identity."""
        raise ValueError("Native ClassVar binding remains unproved")

    def require_generic_alias_construction(self, origin: NativeDeclaration) -> None:
        """Require native origin/argument retention, independently of alias use."""
        raise ValueError("Native generic alias construction remains unproved")

    def dictionary_construction_initial_operand(
        self, node: ast.Dict
    ) -> ast.expr | None:
        """None supplies no narrower application bound than the whole expression."""
        return None

    def require_tuple_construction(self, node: ast.Tuple) -> None:
        """Require retained input assembly, not evaluation, fresh identity or release."""
        raise ValueError("Native tuple construction remains unproved")

    @property
    def primitive_operations(self) -> dict[int, NativePrimitiveOperation]:
        """Unsupported runtimes contribute no native primitive transfer proof."""
        return {}

    def capture_function_operand(
        self,
        stack: NativeOperandStack,
        context: NativeCodeObservation,
        instruction: dis.Instruction,
    ) -> None:
        raise ValueError("Native compiler has no function operand proof")

    def require_object_release(self, value: object) -> None:
        """Require release of an actual captured native object."""
        raise ValueError("Native object lifetime remains unproved")

    def require_inert_instance_release(self, exact_type: type) -> None:
        """Require instance destruction safety independently of contents or identity."""
        raise ValueError("Native instance lifetime remains unproved")

    def require_dictionary_construction(self, node: ast.Dict) -> None:
        """Require construction effects, not operands or resulting object identity."""
        raise ValueError("Native dictionary construction remains unproved")

    def require_empty_dictionary_creation(self, node: ast.Dict) -> None:
        """Require fresh exact-dict production, independently of activation."""
        raise ValueError("Native empty dictionary creation remains unproved")

    def require_dictionary_scalar_store(self, key: NativeScalar) -> None:
        """Require exact dictionary setter behavior under exact scalar-key admission."""
        raise ValueError("Native dictionary item storage remains unproved")

    def class_construction_fields(
        self,
        names: frozenset[NativeScalar],
    ) -> tuple[CPythonClassConstructionField, ...]:
        """Select native construction obligations; actual values discharge them."""
        raise ValueError("Native class construction namespace remains unproved")

    def require_inert_class_member_type(self, value_type: type) -> None:
        """Prove special-hook absence in an immutable native type hierarchy.

        CPython type_new_set_names uses _PyObject_LookupSpecial on each member
        value. It searches the value type's MRO, not the value's attributes.
        Static-type admission also prevents later mutation of the lookup chain.
        No descriptor or analyzed-object attribute lookup is executed here.
        """
        for owner in self.require_static_type_mro(value_type):
            if property.__set_name__.__name__ in type.__getattribute__(
                owner, "__dict__"
            ):
                raise ValueError(
                    "Native class member installation hook remains unproved"
                )

    def require_static_type_release(self, value: object) -> None:
        """Require an immutable static native type with non-deallocating lifetime.

        Consumers may use both its stable type namespace and release safety;
        a retained or immortal object alone does not establish this contract.
        """
        raise ValueError("Native static type lifetime has no admitted runtime proof")

    def class_prologue(self, prefix: NativeCodeObservation) -> NativeClassPrologue:
        return OpenNativeClassPrologue(NativeExecutionUnavailable.UNSUPPORTED_COMPILER)

    def inventory(
        self, code: CodeType, compilation: NativeCompilationIdentity
    ) -> NativeCreationInventory:
        """Observe creations, initial prefixes and entry stores in one native walk."""
        pending = [(code, False)]
        emissions: list[NativeCodeEmission] = []
        builders: list[NativeInstructionSite] = []
        prefixes: dict[int, NativeCodeObservation] = {}
        observations: list[NativeScopeObservation] = []
        while pending:
            parent, requires_prefix = pending.pop()
            has_children = any(
                isinstance(value, CodeType) for value in parent.co_consts
            )
            prefix = NativeScopeObservation(parent, self.primitive_operations)
            prefixes[id(parent)] = prefix
            observations.append(prefix)
            parent_builders: set[SourceByteSpan] = set()
            for instruction in self.instructions(parent):
                prefix.observe(instruction)
                previous = prefix.current
                if has_children:
                    builder = self.builder_load(parent, instruction)
                    if builder is not None:
                        builders.append(builder)
                        if builder.source_span is not None:
                            parent_builders.add(builder.source_span)
                    emission = None
                    if isinstance(instruction.argval, CodeType):
                        emission = NativeCodeEmission(compilation, instruction, parent)
                        pending.append(
                            (
                                emission.code,
                                emission.source_span is not None
                                and emission.source_span in parent_builders,
                            )
                        )
                        emissions.append(emission)
                        if prefix.boundary is None:
                            prefix.emissions.append(emission)
                    prefix.current = self.observe(prefix.current, instruction, emission)
                prefix.observe_values(
                    instruction,
                    previous,
                    after_prologue=not requires_prefix or prefix.boundary is not None,
                )
        return NativeCreationInventory(
            code,
            tuple(emissions),
            tuple(builders),
            prefixes,
            tuple(observations),
        )

    @abstractmethod
    def proves_class_capture(
        self, builder: NativeInstructionSite, body: NativeCodeEmission
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def builder_load(
        self, parent: CodeType, instruction: dis.Instruction
    ) -> NativeInstructionSite | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def class_capture_fallback(self) -> NativeExecutionUnavailable:
        raise NotImplementedError

    @classmethod
    @lru_cache(maxsize=None)
    def annotation_order(cls) -> NativeAnnotationOrder:
        return OpenNativeAnnotationOrder(
            NativeAnnotationOrder.probe_compilation().identity,
            NativeAnnotationOrderUnavailable.UNSUPPORTED_COMPILER,
        )

    @classmethod
    def current(cls) -> NativeCreationBackend:
        native_identity = (sys.implementation.name, sys.version_info[:2])
        return cls.__registry__.get(native_identity, SpanOnlyCreationBackend)()

    @abstractmethod
    def instructions(self, parent: CodeType) -> Iterable[dis.Instruction]:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        raise NotImplementedError

    @abstractmethod
    def proves_body(self, emission: NativeCodeEmission) -> bool:
        raise NotImplementedError

    def select(self, bucket: list[NativeCodeEmission]) -> NativeCodeEmission | None:
        """Keep all emission sites until this single cardinality decision."""
        candidates = (
            bucket
            if len(bucket) == 1
            else tuple(emission for emission in bucket if self.proves_body(emission))
        )
        return candidates[0] if len(candidates) == 1 else None

    def emissions(
        self, code: CodeType, compilation: NativeCompilationIdentity
    ) -> Iterator[NativeCodeEmission]:
        """Project the shared inventory; no proof is selected before it completes."""
        yield from self.inventory(code, compilation).emissions

    def project(
        self, code: CodeType, compilation: NativeCompilationIdentity
    ) -> _NativeExecutionIndex:
        buckets: dict[SourceByteSpan, list[NativeCodeEmission]] = {}
        inventory = self.inventory(code, compilation)
        incomplete = any(site.source_span is None for site in inventory.builder_loads)
        for emission in inventory.emissions:
            span = emission.source_span
            if span is None:
                incomplete = True
            else:
                buckets.setdefault(span, []).append(emission)
        ambiguous: set[SourceByteSpan] = set()
        selected_emissions: list[NativeCodeEmission] = []
        for span, bucket in buckets.items():
            selected = self.select(bucket)
            if selected is None:
                ambiguous.add(span)
            else:
                selected_emissions.append(selected)
        class_captures = inventory.class_captures(
            compilation, tuple(selected_emissions), self
        )
        exact = {
            cast(SourceByteSpan, emission.source_span): cast(
                ExactNativeFunctionExecution, emission.receipt
            )
            for emission in selected_emissions
        }
        return _NativeExecutionIndex(
            compilation,
            IdentityHandleMultiplicityProjection(exact, frozenset(ambiguous)),
            incomplete,
            class_captures,
            self.class_capture_fallback,
            inventory.scopes,
        )


class SpanOnlyCreationBackend(NativeCreationBackend):
    """No compiler-origin body-role claim outside an admitted backend."""

    class_capture_fallback = NativeExecutionUnavailable.UNSUPPORTED_COMPILER

    def proves_class_capture(
        self, builder: NativeInstructionSite, body: NativeCodeEmission
    ) -> bool:
        return False

    def builder_load(self, parent: CodeType, instruction: dis.Instruction) -> None:
        return None

    def instructions(self, parent: CodeType) -> Iterable[dis.Instruction]:
        return dis.get_instructions(parent)

    def observe(
        self,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        return None

    def proves_body(self, emission: NativeCodeEmission) -> bool:
        return False


class NativeAnnotationOrderUnavailable(StrEnum):
    UNSUPPORTED_COMPILER = "unsupported_annotation_compiler"
    COMPILATION_REJECTED = "annotation_probe_rejected"
    INCOMPLETE_EMISSION = "incomplete_annotation_emission"
    NONUNIFORM_GROUPS = "nonuniform_annotation_groups"


@dataclass(frozen=True)
class NativeAnnotationOrder(ABC):
    """Compiler-conditioned annotation-root order, not expression-effect proof."""

    compilation: NativeCompilationIdentity

    @staticmethod
    def probe_compilation() -> NativePythonCompilation:
        return NativePythonCompilation(
            "def probe(a: first, b: second, /, c: third, d: fourth, "
            "*rest: fifth, e: sixth, f: seventh, **kw: eighth) -> ninth: pass\n",
            "<native-annotation-order>",
        )

    def parameter_sources(
        self, arguments: ast.arguments
    ) -> tuple[FunctionArgumentSource, ...]:
        return FunctionArgumentSource.from_arguments(arguments)

    def visit_in(
        self,
        visitor: FunctionAnnotationVisitor,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        roots = FunctionArgumentSource.annotation_roots(
            self.parameter_sources(node.args), node.returns
        )
        self.visit_roots(visitor, roots)

    @abstractmethod
    def visit_roots(
        self, visitor: FunctionAnnotationVisitor, roots: tuple[ast.expr, ...]
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactNativeAnnotationOrder(NativeAnnotationOrder):
    parameter_kinds: tuple[CompactParameterKind, ...]

    def __post_init__(self) -> None:
        if len(self.parameter_kinds) != len(CompactParameterKind) or set(
            self.parameter_kinds
        ) != set(CompactParameterKind):
            raise ValueError("Annotation order must include every parameter kind once")

    def parameter_sources(
        self, arguments: ast.arguments
    ) -> tuple[FunctionArgumentSource, ...]:
        parameters = super().parameter_sources(arguments)
        return tuple(
            parameter
            for kind in self.parameter_kinds
            for parameter in parameters
            if parameter.kind is kind
        )

    def visit_roots(
        self, visitor: FunctionAnnotationVisitor, roots: tuple[ast.expr, ...]
    ) -> None:
        visitor.visit_ordered_annotations(roots)


@dataclass(frozen=True)
class OpenNativeAnnotationOrder(NativeAnnotationOrder):
    reason: NativeAnnotationOrderUnavailable

    def visit_roots(
        self, visitor: FunctionAnnotationVisitor, roots: tuple[ast.expr, ...]
    ) -> None:
        visitor.visit_unordered_annotations(roots)


class EagerAnnotationOrderBackend(NativeCreationBackend):
    """Admitted compiler's uniform kind-group emission, checked without execution."""

    @classmethod
    @lru_cache(maxsize=None)
    def annotation_order(cls) -> NativeAnnotationOrder:
        compilation = NativeAnnotationOrder.probe_compilation()
        try:
            code = compilation.compile()
        except SyntaxError:
            return OpenNativeAnnotationOrder(
                compilation.identity,
                NativeAnnotationOrderUnavailable.COMPILATION_REJECTED,
            )
        function = ast.parse(compilation.source).body[0]
        parameters = FunctionArgumentSource.from_arguments(function.args)
        roots = tuple(parameter.argument.annotation for parameter in parameters) + (
            function.returns,
        )
        offsets = UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                instruction
                for instruction in cls().instructions(code)
                if None not in instruction.positions
            ),
            lambda instruction: SourceByteSpan(
                instruction.positions.lineno - 1,
                instruction.positions.end_lineno - 1,
                instruction.positions.col_offset,
                instruction.positions.end_col_offset,
            ),
        )
        spans = tuple(SourceByteSpan.require_node(root) for root in roots)
        if any(span not in offsets for span in spans):
            return OpenNativeAnnotationOrder(
                compilation.identity,
                NativeAnnotationOrderUnavailable.INCOMPLETE_EMISSION,
            )
        order = tuple(
            sorted(range(len(spans)), key=lambda index: offsets[spans[index]].offset)
        )
        groups = tuple(
            dict.fromkeys(
                parameters[index].kind for index in order if index < len(parameters)
            )
        )
        expected = tuple(
            index
            for kind in groups
            for index, parameter in enumerate(parameters)
            if parameter.kind is kind
        ) + (len(parameters),)
        if order != expected or set(groups) != set(CompactParameterKind):
            return OpenNativeAnnotationOrder(
                compilation.identity, NativeAnnotationOrderUnavailable.NONUNIFORM_GROUPS
            )
        return ExactNativeAnnotationOrder(compilation.identity, groups)


class ContiguousNativeCreationBackend(NativeCreationBackend):
    """Admitted native creation transfers shared by supported CPython backends."""

    class_capture_fallback = NativeExecutionUnavailable.NO_EMITTED_CODE
    attribute_flags: frozenset[int]
    application_prelude: tuple[type[NativeCreationOperation], ...]

    def require_invocation(
        self, prelude: list[dis.Instruction], call: dis.Instruction
    ) -> None:
        if len(prelude) != len(self.application_prelude):
            raise ValueError("Native invocation has a different preparation protocol")
        span = NativeInstructionSite.span_for(call)
        if span is None or type(call.arg) is not int or call.arg < 0:
            raise ValueError(
                "Native invocation lacks its source range or argument count"
            )
        for instruction, declaration in zip(
            prelude, self.application_prelude, strict=True
        ):
            if instruction.opcode != dis.opmap[declaration.native_name]:
                raise ValueError(
                    "Native invocation preparation is not its declared operation"
                )
        for instruction in (*prelude, call):
            if (
                type(instruction.arg) is not int
                or instruction.arg != call.arg
                or instruction.is_jump_target
                or NativeInstructionSite.span_for(instruction) != span
            ):
                raise ValueError(
                    "Native invocation preparation differs from its original call"
                )

    def require_definition_application(
        self, prelude: list[dis.Instruction], call: dis.Instruction
    ) -> None:
        self.require_invocation(prelude, call)
        if call.arg != 0:
            raise ValueError(
                "Native definition application requires zero explicit arguments"
            )

    @cached_property
    def primitive_operations(self) -> dict[int, NativePrimitiveOperation]:
        return {
            dis.opmap[operation.name]: operation
            for operation in NativePrimitiveOperation
            if operation.name in dis.opmap
        }

    def capture_function_operand(
        self,
        stack: NativeOperandStack,
        context: NativeCodeObservation,
        instruction: dis.Instruction,
    ) -> None:
        operation = self.operations.get(instruction.opcode)
        if operation is None:
            raise ValueError("Native function operand operation remains unproved")
        emission = context.require_current_creation(instruction)
        operation.capture_operands(stack, emission, instruction)

    def class_prologue(self, prefix: NativeCodeObservation) -> NativeClassPrologue:
        initial = prefix.initial_instructions
        if not initial:
            return OpenNativeClassPrologue(
                NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES
            )
        stack = NativeOperandStack(code=prefix.code)
        for instruction in initial:
            operation = self.primitive_operations.get(instruction.opcode)
            if instruction.is_jump_target:
                return OpenNativeClassPrologue(
                    NativeExecutionUnavailable.UNSUPPORTED_PROLOGUE
                )
            if operation is None:
                creation = self.operations.get(instruction.opcode)
                if creation is None:
                    return OpenNativeClassPrologue(
                        NativeExecutionUnavailable.UNSUPPORTED_PROLOGUE
                    )
                try:
                    creation.capture_prologue(stack, prefix, instruction)
                except ValueError:
                    return OpenNativeClassPrologue(
                        NativeExecutionUnavailable.UNSUPPORTED_PROLOGUE
                    )
                continue
            if None in instruction.positions and not operation.unpositioned_setup:
                return OpenNativeClassPrologue(
                    NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES
                )
            try:
                completion = operation.capture(stack, instruction)
            except ValueError:
                return OpenNativeClassPrologue(
                    NativeExecutionUnavailable.UNSUPPORTED_PROLOGUE
                )
            if completion is not None:
                return completion
        if prefix.boundary is None:
            return OpenNativeClassPrologue(
                NativeExecutionUnavailable.INCOMPLETE_SOURCE_RANGES
            )
        return stack.complete(prefix.boundary.offset)

    @cached_property
    def operations(self) -> dict[int, type[NativeCreationOperation]]:
        return {
            dis.opmap[operation.native_name]: operation
            for operation in NativeCreationOperation.__registry__.values()
            if operation.native_name in dis.opmap
        }

    @cached_property
    def builder_opcode(self) -> int:
        return dis.opmap[NativePrimitiveOperation.LOAD_BUILD_CLASS.name]

    def builder_load(
        self, parent: CodeType, instruction: dis.Instruction
    ) -> NativeInstructionSite | None:
        if instruction.opcode != self.builder_opcode:
            return None
        return NativeInstructionSite(parent, instruction)

    def proves_class_capture(
        self, builder: NativeInstructionSite, body: NativeCodeEmission
    ) -> bool:
        """Admit original CPython class lowering, not arbitrary bytecode.

        LOAD_BUILD_CLASS is emitted for the implicit class statement operation,
        and its ClassDef range covers the raw body creation in the same code.
        The inventory first requires unique builder and child-code load sites
        in that exact range/container. The observed MAKE_FUNCTION then supplies
        actual creation, including for a body inside a generated wrapper.
        These are two source-origin captures, NOT proof of their runtime values,
        stack argument binding, body entry or absence of header effects.
        """
        return (
            builder.containing_code is body.containing_code
            and builder.source_span is not None
            and body.source_span is not None
            and builder.source_span == body.source_span
            and body.creation is not None
            and NativeInstructionSite(body.containing_code, body.creation).source_span
            is not None
            and builder.instruction.offset < body.load.offset < body.creation.offset
        )

    def instructions(self, parent: CodeType) -> Iterable[dis.Instruction]:
        return dis.Bytecode(parent)

    def observe(
        self,
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        if instruction.is_jump_target:
            current = None
        operation = self.operations.get(instruction.opcode)
        if operation is None:
            return None
        return operation.advance(self, current, instruction, emission)


class CPythonClassConstructionField(StrEnum):
    """Native ABI fields own their operand obligations and activation witnesses.

    Each declared invalid operand is inert and rejected when that native role
    exists. Neither its probe class nor its values stand in for target objects.
    Actual captured values discharge the selected operation independently.
    """

    def _require_cell_or_layout(
        self,
        value: CapturedReferenceResolution,
        namespace: NamespaceCreationEvidenceABC,
    ) -> None:
        value.require_class_construction_field(self, namespace)

    def _require_documentation(
        self,
        value: CapturedReferenceResolution,
        namespace: NamespaceCreationEvidenceABC,
    ) -> None:
        value.require_closed()
        if issubclass(value.native_type, str):
            value.require_native_text().encode("utf-8")

    def __new__(
        cls,
        name: str,
        validator: Callable[
            [
                CPythonClassConstructionField,
                CapturedReferenceResolution,
                NamespaceCreationEvidenceABC,
            ],
            None,
        ] = _require_cell_or_layout,
        invalid_operand: object = object(),
    ) -> Self:
        member = str.__new__(cls, name)
        member._value_ = name
        member.validator = validator
        member.invalid_operand = invalid_operand
        return member

    SLOTS = "__slots__"
    CLASS_CELL = "__classcell__"
    CLASS_DICTIONARY_CELL = "__classdictcell__"
    DOCUMENTATION = "__doc__", _require_documentation, "\ud800"

    @cached_property
    def native_active(self) -> bool:
        try:
            type(self.name, (), {self.value: self.invalid_operand})
        except (TypeError, UnicodeError):
            return True
        return False

    def require_value(
        self,
        value: CapturedReferenceResolution,
        namespace: NamespaceCreationEvidenceABC,
    ) -> None:
        self.validator(self, value, namespace)

    def require_default(self) -> None:
        raise ValueError(f"Native class construction for {self.value} remains unproved")


class CPythonClassConstruction(NativeCreationBackend):

    # These exact native getters implement ordinary attribute lookup. The
    # declaration is about lookup behavior, not membership in a scalar domain.
    # No analyzed instance, descriptor or user-defined getter is invoked.
    ordinary_member_getters = tuple(
        type.__getattribute__(owner, "__getattribute__")
        for owner in (object, str, int, bool, tuple, list, dict)
    )
    abstract_member_marker = "__isabstractmethod__"

    def require_nonabstract_member_type(self, value_type: type) -> None:
        self.require_static_type_mro(value_type)
        declaration = NativeClassMroDeclaration(value_type)
        getter = type.__getattribute__(value_type, "__getattribute__")
        if not any(getter is original for original in self.ordinary_member_getters):
            raise ValueError("Native abstract-member getter remains unproved")
        if declaration.member_owner("__getattr__") is not None or (
            declaration.member_owner(self.abstract_member_marker) is not None
        ):
            raise ValueError("Native abstract-member marker lookup remains unproved")

    def require_nonabstract_member(self, value_type: type) -> None:
        self.require_nonabstract_member_type(value_type)
        if type.__getattribute__(value_type, "__dictoffset__") != 0:
            raise ValueError("Native abstract-member instance storage remains unproved")

    def require_fresh_class_namespace(
        self, metaclass: NativeClassMroDeclaration
    ) -> None:
        """Apply type_prepare's PyDict_New law after authenticating ordinary lookup."""
        # The actual selected descriptor is inspected, never invoked. The source
        # owner proves metaclass selection, header completion and behavior at its
        # activation; this law supplies no observed namespace or later class result.
        if self.compiler_identity != (sys.implementation.name, sys.version_info[:2]):
            raise ValueError(
                "Fresh class namespace requires its actual native interpreter backend"
            )
        metaclass.require_type_preparation()

    def class_construction_fields(
        self,
        names: frozenset[NativeScalar],
    ) -> tuple[CPythonClassConstructionField, ...]:
        return tuple(
            requirement
            for requirement in CPythonClassConstructionField
            if requirement.value in names and requirement.native_active
        )


class CPythonFunctionConstruction(NativeCreationBackend):
    """Native function birth-state law shared by the admitted CPython backends.

    Objects/funcobject.c initializes func_dict to NULL. MAKE_FUNCTION and
    permitted native metadata attachments do not populate custom attributes.
    This is conditional on noninterfering external instrumentation and native
    API compliance; source activation must supply that execution restriction.
    Invoked source/dependency callbacks remain separate operation obligations.
    No later attribute state, abstractness lookup or callable behavior is proved.
    """

    def require_fresh_function_namespace(
        self, compilation: NativeCompilationIdentity
    ) -> None:
        if compilation.interpreter != (
            sys.implementation.name,
            sys.version,
        ) or self.compiler_identity != (sys.implementation.name, sys.version_info[:2]):
            raise ValueError(
                "Fresh function namespace requires its actual native interpreter backend"
            )


class CPythonStaticTypeRequirement(IntEnum):
    """Native flag requirements for supported CPython static type lifetime."""

    IMMUTABLETYPE = (1 << 8, True)
    HEAPTYPE = (1 << 9, False)

    def __new__(cls, mask: int, required: bool) -> Self:
        member = int.__new__(cls, mask)
        member._value_ = mask
        member.required = required
        return member

    def require(self, flags: int) -> None:
        if bool(flags & self.value) is not self.required:
            raise ValueError("Native static type lifetime remains unproved")


class CPythonValueLifetime:
    """Release contracts for the supported CPython native object implementations.

    Static type objects retain the existing immutable/non-heap ABI proof.
    Exact Unicode values have no user-owned referents or weakref storage;
    unicode_dealloc only manages native buffers and the native intern table.
    None and bool singleton lifetimes are indestructible in 3.11 and immortal
    in 3.14 under the backend's reference-ownership premise. Exact bool cannot
    be subclassed and carries no user-owned references or weakref storage.
    These contracts come from Objects/unicodeobject.c, Objects/object.c and
    Objects/boolobject.c at the supported CPython versions,
    not from analyzer-held references or an assumed target constant identity.
    Exact integers likewise own only native digits: 3.11 inherits object deallocation
    and 3.14 long_dealloc/its native freelist, with no instance dictionary,
    weakrefs or user-owned referents (Objects/longobject.c). Integer subclasses
    and other exact types retain an unresolved instance proof. Exact object
    instances have no dictionary, weakrefs or owned Python references;
    object_dealloc forwards to the native free slot (Objects/typeobject.c).
    """

    inert_instance_types: ClassVar[tuple[type, ...]] = (
        str,
        type(None),
        bool,
        int,
        object,
    )

    @classmethod
    def require_inert_instance_release(cls, exact_type: type) -> None:
        if not any(exact_type is declared for declared in cls.inert_instance_types):
            raise ValueError("Native instance lifetime remains unproved")

    @classmethod
    def require_object_release(cls, value: object) -> None:
        if type(value) is type:
            cls.require_static_type_release(value)
        else:
            cls.require_inert_instance_release(type(value))

    @staticmethod
    def require_static_type_release(value: object) -> None:
        if type(value) is not type:
            raise ValueError("Native static type lifetime remains unproved")
        flags = type.__getattribute__(value, "__flags__")
        for requirement in CPythonStaticTypeRequirement:
            requirement.require(flags)


class CPythonContainerConstruction(NativeCreationBackend):
    """Conditional native-container construction in the admitted compilers.

    compiler_dict/codegen_dict create exact native dictionaries, including
    internal chunks. Literal key hashing/equality cannot invoke user code.
    Distinct keys cannot overwrite entries, independent of chunk scheduling;
    collisions additionally require the displaced value's release contract.
    Unpacked mappings and operand evaluation retain separate obligations.

    Native sources: CPython v3.11.11 Python/compile.c compiler_dict and
    v3.14.0 Python/codegen.c codegen_dict. No bytecode timing is inferred here.
    """

    generic_alias_origins: ClassVar[tuple[NativeDeclaration, ...]] = tuple(
        NativeDeclaration(origin)
        for origin in (dict, enumerate, frozenset, list, set, tuple, BaseExceptionGroup)
    )

    def require_generic_alias_construction(self, origin: NativeDeclaration) -> None:
        """Supported CPython builtin slots call Py_GenericAlias directly.

        setup_ga retains origin and the original tuple argument, or packs one
        argument without invoking its hooks. Hashing, metadata, release and base
        expansion are separate operations. This is the native implementation law
        in Objects/genericaliasobject.c for CPython 3.11.11 and 3.14.0, not an
        inference from a __class_getitem__ spelling or descriptor shape.
        """
        if origin not in self.generic_alias_origins:
            raise ValueError("Native origin has no admitted generic alias constructor")
        self.require_static_type_release(origin.declaration)

    def dictionary_construction_initial_operand(
        self, node: ast.Dict
    ) -> ast.expr | None:
        """Native map insertion starts no earlier than its first value evaluation."""
        # compiler_subdict/codegen_subdict evaluate the initial value before
        # MAP_ADD, nonempty BUILD_MAP or BUILD_CONST_KEY_MAP. Earlier empty map
        # allocation has no key/value protocol effects. Unknown first-key
        # evaluation retains the whole-expression bound.
        if not isinstance(node, ast.Dict):
            raise ValueError(
                "Dictionary construction requires original dictionary syntax"
            )
        if not node.keys or node.keys[0] is None:
            return None
        try:
            LiteralExpressionEffects(node.keys[0]).hashable_value
        except ValueError:
            return None
        return node.values[0]

    def require_tuple_construction(self, node: ast.Tuple) -> None:
        """CPython compiler_tuple/codegen_tuple retain non-starred inputs in order.

        Constant folding and large-list assembly preserve that ordered content.
        Input evaluation and lifetimes remain separate; no hashing or source
        iterator is invoked by this non-unpacking construction law. This does
        not imply fresh identity or inert element release.
        """
        if not isinstance(node, ast.Tuple) or not isinstance(node.ctx, ast.Load):
            raise ValueError("Tuple construction requires original load syntax")
        if any(isinstance(element, ast.Starred) for element in node.elts):
            raise ValueError("Unpacked tuple construction remains unproved")

    def require_dictionary_construction(self, node: ast.Dict) -> None:
        retained_values: dict[Hashable, ast.expr] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise ValueError("Unpacked mapping construction remains unproved")
            key = LiteralExpressionEffects(key_node).hashable_value
            if key in retained_values:
                previous = LiteralExpressionEffects(retained_values[key]).value
                self.require_inert_instance_release(type(previous))
            retained_values[key] = value_node

    def require_empty_dictionary_creation(self, node: ast.Dict) -> None:
        """Admitted compiler_dict/codegen_dict emit a fresh exact empty dictionary.

        No key, value, hash, equality or release operation occurs in this form.
        This conditional compiler law does not establish that source ran once.
        """
        if not isinstance(node, ast.Dict) or node.keys or node.values:
            raise ValueError(
                "Fresh empty dictionary requires its original empty syntax"
            )
        self.require_dictionary_construction(node)

    def require_dictionary_scalar_store(self, key: NativeScalar) -> None:
        """Exact dict storage retains the RHS and cannot run a custom setter.

        All resident keys must remain admitted exact scalars under the
        namespace/prefix proof. Their hash/equality cannot invoke user protocols.
        Temporary-key release uses the existing native lifetime authority.
        Old-value and temporary-receiver release remain separate obligations.
        """
        if not NativeScalarValueABC.supports_scalar(key):
            raise ValueError("Native dictionary stores require an exact scalar key")
        self.require_inert_instance_release(type(key))


class CPythonTypingConstruction(NativeCreationBackend):
    """Supported standard-library binding on certified inert representatives.

    ClassVar's native implementation performs _type_check and ForwardRef parsing.
    Preserve its value-dependent acceptance and version-specific rules by using
    that implementation, without interpreting source expressions or forward refs.
    This relies on the admitted unmodified standard library, just like the native
    declaration premise; source-side mutations remain original-prefix obligations.
    The uncached binding body avoids unrelated analyzer cache-key callbacks.
    Target cache completion/integrity remains a separate obligation; this
    probe supplies neither cached target result nor base-origin evidence.
    """

    class_variable = NativeDeclaration(ClassVar)

    def require_classvar_binding(self, argument: InertNativeArgumentWitness) -> None:
        value = argument.value
        declaration = self.class_variable.declaration
        try:
            declaration._getitem(declaration, value)
        except (TypeError, SyntaxError) as error:
            raise ValueError("Native ClassVar argument binding rejected") from error


class CPython311CreationBackend(
    CPythonValueLifetime,
    EagerAnnotationOrderBackend,
    ContiguousNativeCreationBackend,
    SpanOnlyCreationBackend,
    CPythonClassConstruction,
    CPythonFunctionConstruction,
    CPythonContainerConstruction,
    CPythonTypingConstruction,
):
    compiler_identity = ("cpython", (3, 11))
    attribute_flags = frozenset()
    application_prelude = (PrepareNativeDefinitionApplication,)

    call_operand_order = NativeCallOperandOrder.NULL_CALLEE


class CPython314CreationBackend(
    CPythonValueLifetime,
    ContiguousNativeCreationBackend,
    CPythonClassConstruction,
    CPythonFunctionConstruction,
    CPythonContainerConstruction,
    CPythonTypingConstruction,
):
    """Admit CPython 3.14's annotation attachment to its raw function body.

    This is a compiler construction invariant, not opcode-shape inference:
    codegen_function_annotations supplies MAKE_FUNCTION_ANNOTATE to
    codegen_function_body. Generated providers/wrappers lack that attachment.
    The preserved target is the raw body, before decorators or wrapper calls.
    """

    compiler_identity = ("cpython", (3, 14))
    application_prelude = ()

    call_operand_order = NativeCallOperandOrder.CALLEE_NULL

    def __init__(self) -> None:
        # Access only after explicit compiler admission; missing capability is
        # a failed backend contract, never an empty metadata fallback.
        self.native_attributes: tuple[str, ...] = dis.FUNCTION_ATTR_FLAGS
        if self.native_attributes.count("annotate") != 1:
            raise RuntimeError(
                "Admitted CPython 3.14 backend lacks unique annotate metadata"
            )

    @cached_property
    def binary_subscription_argument(self) -> int:
        # This is the admitted interpreter ABI table, not an analyzer operator map.
        return tuple(name for name, symbol in dis._nb_ops).index("NB_SUBSCR")

    @cached_property
    def annotate_flag(self) -> int:
        return 1 << self.native_attributes.index("annotate")

    @cached_property
    def attribute_flags(self) -> frozenset[int]:
        return frozenset(1 << index for index, _ in enumerate(self.native_attributes))

    def proves_body(self, emission: NativeCodeEmission) -> bool:
        return any(event.arg == self.annotate_flag for event in emission.attachments)


@dataclass(frozen=True)
class NativePythonCompilation:
    """Lazily project original module context without importing or executing it.

    Only compact receipts are cached; native executable code is transient.
    """

    source: str
    file_path: str

    @ScanCache.cached
    def _function_source_span(
        self, code_contents: tuple[bytes, tuple[int, ...]], filename: str
    ) -> SourceByteSpan:
        """Cache source correspondence by complete code contents within one scan.

        Marshal format 2 excludes intern/reference-table layout, while retaining
        native code metadata and nested code. Plain code equality omits observable
        metadata. Neither target execution nor executable-code retention is needed.
        """
        backend = NativeCreationBackend.current()
        emissions = tuple(
            backend.emissions(self.compile(filename=filename), self.identity)
        )
        matches = tuple(
            emission
            for emission in emissions
            if backend.code_contents(emission.code) == code_contents
        )
        if len(matches) != 1:
            raise ValueError("Python implementation differs from compiled source")
        (matched,) = matches
        span = matched.source_span
        if (
            span is None
            or matched.creation is None
            or backend.select(
                [emission for emission in emissions if emission.source_span == span]
            )
            is not matched
        ):
            raise ValueError("Python implementation has no unique compiler body")
        return span

    def function_definition(
        self, function: FunctionType
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Match current code contents, not function identity or runtime behavior.

        Defaults, closure values, globals, activation and effects need independent
        evidence. A fresh syntax tree prevents callers from modifying cached proof.
        """
        if type(function) is not FunctionType:
            raise ValueError("Python implementation requires an exact function")
        code = function.__code__
        span = self._function_source_span(
            NativeCreationBackend.current().code_contents(code), code.co_filename
        )
        definitions = tuple(
            node
            for node in ast.walk(ast.parse(self.source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and SourceByteSpan.require_node(node) == span
        )
        if len(definitions) != 1:
            raise ValueError("Native body has no unique function source declaration")
        return definitions[0]

    @classmethod
    def from_function(cls, function: FunctionType) -> Self:
        """Read the complete defining source without invoking the function."""
        if type(function) is not FunctionType:
            raise ValueError("Python implementation requires an exact function")
        try:
            lines, _ = inspect.findsource(function)
        except (OSError, TypeError, IndexError) as error:
            raise ValueError(
                "Python implementation has no inspectable source"
            ) from error
        return cls("".join(lines), function.__code__.co_filename)

    @property
    def module_annotation_setup(self) -> NativeValueStore | None:
        return self.execution_outcome.module_annotation_setup

    def return_after_binding(self, span: SourceByteSpan, name: str) -> NativeReturn:
        return self.execution_outcome.return_after_binding(span, name)

    def prologue_return_operand(
        self,
        capture: ExactNativeClassCapture,
        receipt: NativeReturn,
        value: NativeProducedValue,
    ) -> NativeProducedValue:
        return self.execution_outcome.prologue_return_operand(capture, receipt, value)

    def value_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        return self.execution_outcome.value_store_for(
            production_span, target_span, name
        )

    def return_after_effect(
        self, span: SourceByteSpan, declaration: type[NativeStackEffectABC]
    ) -> NativeReturn:
        return self.execution_outcome.return_after_effect(span, declaration)

    def return_from(self, execution: NativeFunctionExecution) -> NativeReturn:
        return self.execution_outcome.return_from(execution)

    def return_after(self, execution: NativeFunctionExecution) -> NativeReturn:
        return self.execution_outcome.return_after(execution)

    def scalar_store_for(
        self, production_span: SourceByteSpan, target_span: SourceByteSpan, name: str
    ) -> NativeValueStore:
        return self.execution_outcome.scalar_store_for(
            production_span, target_span, name
        )

    def require_fresh_function_namespace(
        self, receipt: NativeFunctionExecution
    ) -> None:
        """Authenticate one original raw creation before applying its native birth law."""
        # Conditional initialization is not an observed target function or continued
        # emptiness through decorators, later writes or callbacks. Runtime activation
        # and noninterference remain the source execution owner's duties.
        self.execution_outcome.require_function(receipt)
        receipt.require_creation()
        NativeCreationBackend.current().require_fresh_function_namespace(self.identity)

    def constant_store_for(
        self, source_span: SourceByteSpan, name: str
    ) -> NativeConstantStore:
        return self.execution_outcome.constant_store_for(source_span, name)

    def class_capture_for(self, source_span: SourceByteSpan) -> NativeClassCapture:
        return self.execution_outcome.class_capture_for(source_span)

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", source_path_text(self.file_path))

    @cached_property
    def identity(self) -> NativeCompilationIdentity:
        return NativeCompilationIdentity(
            self.file_path,
            python_source_cache_signature(self.source),
            (sys.implementation.name, sys.version),
        )

    def compile(self, *, filename: str | None = None) -> CodeType:
        """Return transient native code, preserving compiler errors for validation."""
        if filename is None:
            filename = self.file_path
        if source_path_text(filename) != self.file_path:
            raise ValueError("Compiler filename belongs to a different source path")
        return compile(self.source, filename, "exec", dont_inherit=True, optimize=0)

    @property
    def execution_outcome(self) -> _NativeCompilationOutcome:
        """Expose only the native observation belonging to this source owner."""
        outcome = self._execution_outcome
        if outcome.compilation is not self.identity:
            raise ValueError("Native outcome belongs to a different compilation")
        return outcome

    @cached_property
    def _execution_outcome(self) -> _NativeCompilationOutcome:
        try:
            return NativeCreationBackend.current().project(
                self.compile(), self.identity
            )
        except SyntaxError:
            return _RejectedNativeCompilation(self.identity)

    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        return self.execution_outcome.execution_for(source_span)
