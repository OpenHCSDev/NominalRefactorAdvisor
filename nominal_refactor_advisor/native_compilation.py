"""Native raw-code execution evidence without executing analyzed modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import dis
from enum import IntEnum, StrEnum
from functools import cached_property
import inspect
import sys
from types import CodeType
from typing import Self

from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
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

    @classmethod
    def from_code(cls, code: CodeType, compilation: NativeCompilationIdentity) -> Self:
        creations: list[ExactNativeFunctionExecution] = []
        pending = [code]
        has_incomplete_ranges = False
        while pending:
            parent = pending.pop()
            for instruction in dis.get_instructions(parent):
                child = instruction.argval
                if not isinstance(child, CodeType):
                    continue
                pending.append(child)
                line, end_line, column, end_column = instruction.positions
                if (
                    line is None
                    or end_line is None
                    or column is None
                    or end_column is None
                ):
                    has_incomplete_ranges = True
                    continue
                creations.append(
                    ExactNativeFunctionExecution(
                        compilation,
                        SourceByteSpan(line - 1, end_line - 1, column, end_column),
                        child.co_flags,
                    )
                )
        return cls(
            compilation,
            UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
                creations, lambda creation: creation.source_span
            ),
            has_incomplete_ranges,
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
            return _NativeExecutionIndex.from_code(self.compile(), self.identity)
        except SyntaxError:
            return _RejectedNativeCompilation(self.identity)

    def execution_for(self, source_span: SourceByteSpan) -> NativeFunctionExecution:
        return self._execution_outcome.execution_for(source_span)
