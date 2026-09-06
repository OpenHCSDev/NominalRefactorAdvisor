"""Native raw-code execution evidence without executing analyzed modules."""

from __future__ import annotations

import ast

from abc import ABC, abstractmethod
from collections.abc import (
    Iterable,
    Iterator,
)
from dataclasses import (
    dataclass,
    field,
)
import dis
from enum import IntEnum, StrEnum
from functools import (
    cached_property,
    lru_cache,
)
import inspect
import sys
from types import CodeType
from typing import (
    ClassVar,
    Self,
    cast,
)

from metaclass_registry import AutoRegisterMeta

from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
)
from .lexical_bindings import (
    CompactParameterKind,
    FunctionAnnotationVisitor,
    FunctionArgumentSource,
)
from .source_geometry import SourceByteSpan
from .source_identity import (
    python_source_cache_signature,
    source_path_text,
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
class OpenNativeFunctionExecution(NativeFunctionExecution):
    """The native compiler did not provide a unique positioned code object."""

    reason: NativeExecutionUnavailable

    @property
    def mode(self) -> None:
        return None

    @property
    def violation(self) -> NativeExecutionUnavailable:
        return self.reason


@dataclass(frozen=True)
class _NativeCompilationOutcome(ABC):
    compilation: NativeCompilationIdentity

    @abstractmethod
    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        raise NotImplementedError


class _RejectedNativeCompilation(_NativeCompilationOutcome):
    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        return OpenNativeFunctionExecution(
            self.compilation,
            source_span,
            NativeExecutionUnavailable.COMPILATION_REJECTED,
        )


@dataclass(frozen=True)
class _NativeExecutionIndex(_NativeCompilationOutcome):
    declarations: IdentityHandleMultiplicityProjection[
        SourceByteSpan, ExactNativeFunctionExecution
    ]
    has_incomplete_ranges: bool

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
    All fields are transient. The compact receipt is constructed only once.
    """

    compilation: NativeCompilationIdentity
    load: dis.Instruction
    creation: dis.Instruction | None = None
    attachments: list[dis.Instruction] = field(default_factory=list)

    @property
    def code(self) -> CodeType:
        return cast(CodeType, self.load.argval)

    @cached_property
    def receipt(self) -> ExactNativeFunctionExecution | None:
        line, end_line, column, end_column = self.load.positions
        if None in (line, end_line, column, end_column):
            return None
        return ExactNativeFunctionExecution(
            self.compilation,
            SourceByteSpan(line - 1, end_line - 1, column, end_column),
            self.code.co_flags,
        )


class NativeCreationOperation(ABC, metaclass=AutoRegisterMeta):
    """Three admitted native transfers, registered from their declarations."""

    __registry__: ClassVar[dict[str, type[NativeCreationOperation]]] = {}
    __registry_key__ = "native_name"
    __skip_if_no_key__ = True
    native_name: ClassVar[str]

    @classmethod
    @abstractmethod
    def advance(
        cls,
        attribute_flags: frozenset[int],
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        raise NotImplementedError


class LoadNativeCode(NativeCreationOperation):
    native_name = "LOAD_CONST"

    @classmethod
    def advance(
        cls,
        attribute_flags: frozenset[int],
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        return emission


class MakeNativeFunction(NativeCreationOperation):
    native_name = "MAKE_FUNCTION"

    @classmethod
    def advance(
        cls,
        attribute_flags: frozenset[int],
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        if current is None or current.creation is not None:
            return None
        current.creation = instruction
        return current


class AttachNativeFunctionAttribute(NativeCreationOperation):
    native_name = "SET_FUNCTION_ATTRIBUTE"

    @classmethod
    def advance(
        cls,
        attribute_flags: frozenset[int],
        current: NativeCodeEmission | None,
        instruction: dis.Instruction,
        emission: NativeCodeEmission | None,
    ) -> NativeCodeEmission | None:
        if (
            current is None
            or current.creation is None
            or instruction.arg not in attribute_flags
        ):
            return None
        current.attachments.append(instruction)
        return current


class NativeCreationBackend(ABC, metaclass=AutoRegisterMeta):
    """Compiler-owned role evidence with a shared complete-emission inventory."""

    __registry__: ClassVar[
        dict[tuple[str, tuple[int, int]], type[NativeCreationBackend]]
    ] = {}
    __registry_key__ = "compiler_identity"
    __skip_if_no_key__ = True
    compiler_identity: ClassVar[tuple[str, tuple[int, int]]]

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
        """Stream transient sites; attachment evidence finalizes on exhaustion.

        This is an internal collection phase, not a stream of completed proofs.
        project() exhausts it before selecting any complete source-span bucket.
        """
        pending = [code]
        while pending:
            parent = pending.pop()
            if not any(isinstance(value, CodeType) for value in parent.co_consts):
                continue
            current = None
            for instruction in self.instructions(parent):
                emission = None
                if isinstance(instruction.argval, CodeType):
                    emission = NativeCodeEmission(compilation, instruction)
                    # Merely occurring in co_consts is not emission evidence.
                    pending.append(emission.code)
                    yield emission
                current = self.observe(current, instruction, emission)

    def project(
        self, code: CodeType, compilation: NativeCompilationIdentity
    ) -> _NativeExecutionIndex:
        buckets: dict[SourceByteSpan, list[NativeCodeEmission]] = {}
        incomplete = False
        for emission in self.emissions(code, compilation):
            receipt = emission.receipt
            if receipt is None:
                incomplete = True
            else:
                buckets.setdefault(receipt.source_span, []).append(emission)
        exact: dict[SourceByteSpan, ExactNativeFunctionExecution] = {}
        ambiguous: set[SourceByteSpan] = set()
        for span, bucket in buckets.items():
            selected = self.select(bucket)
            if selected is None:
                ambiguous.add(span)
            else:
                # Buckets only contain emissions with an existing exact receipt.
                exact[span] = cast(ExactNativeFunctionExecution, selected.receipt)
        return _NativeExecutionIndex(
            compilation,
            IdentityHandleMultiplicityProjection(exact, frozenset(ambiguous)),
            incomplete,
        )


class SpanOnlyCreationBackend(NativeCreationBackend):
    """No compiler-origin body-role claim outside an admitted backend."""

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


class CPython311CreationBackend(EagerAnnotationOrderBackend, SpanOnlyCreationBackend):
    compiler_identity = ("cpython", (3, 11))


class CPython314CreationBackend(NativeCreationBackend):
    """Admit CPython 3.14's annotation attachment to its raw function body.

    This is a compiler construction invariant, not opcode-shape inference:
    codegen_function_annotations supplies MAKE_FUNCTION_ANNOTATE to
    codegen_function_body. Generated providers/wrappers lack that attachment.
    The preserved target is the raw body, before decorators or wrapper calls.
    """

    compiler_identity = ("cpython", (3, 14))

    def __init__(self) -> None:
        # Access only after explicit compiler admission; missing capability is
        # a failed backend contract, never an empty metadata fallback.
        self.native_attributes: tuple[str, ...] = dis.FUNCTION_ATTR_FLAGS
        if self.native_attributes.count("annotate") != 1:
            raise RuntimeError(
                "Admitted CPython 3.14 backend lacks unique annotate metadata"
            )

    @cached_property
    def annotate_flag(self) -> int:
        return 1 << self.native_attributes.index("annotate")

    @cached_property
    def attribute_flags(self) -> frozenset[int]:
        return frozenset(1 << index for index, _ in enumerate(self.native_attributes))

    @cached_property
    def operations(self) -> dict[int, type[NativeCreationOperation]]:
        return {
            dis.opmap[operation.native_name]: operation
            for operation in NativeCreationOperation.__registry__.values()
        }

    def instructions(self, parent: CodeType) -> Iterable[dis.Instruction]:
        # Includes direct jumps plus native exception entries and boundaries.
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
        return operation.advance(self.attribute_flags, current, instruction, emission)

    def proves_body(self, emission: NativeCodeEmission) -> bool:
        return any(event.arg == self.annotate_flag for event in emission.attachments)


@dataclass(frozen=True)
class NativePythonCompilation:
    """Lazily project original module context without importing or executing it.

    Only compact receipts are cached; native executable code is transient.
    """

    source: str
    file_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", source_path_text(self.file_path))

    @cached_property
    def identity(self) -> NativeCompilationIdentity:
        return NativeCompilationIdentity(
            self.file_path,
            python_source_cache_signature(self.source),
            (sys.implementation.name, sys.version),
        )

    def compile(self) -> CodeType:
        """Return transient native code, preserving compiler errors for validation."""
        return compile(
            self.source, self.file_path, "exec", dont_inherit=True, optimize=0
        )

    @cached_property
    def _execution_outcome(self) -> _NativeCompilationOutcome:
        try:
            return NativeCreationBackend.current().project(
                self.compile(), self.identity
            )
        except SyntaxError:
            return _RejectedNativeCompilation(self.identity)

    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        return self._execution_outcome.execution_for(source_span)
