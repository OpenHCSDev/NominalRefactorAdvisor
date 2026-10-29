"""Whose field a foreign absence probe reads, and how that owner was found.

A probe `receiver.field is None` restates a state of the receiver's class. The
owner is that class. Per module the receiver is recorded as its declared
receiver path (`DeclaredPathProbeOwner`, `declared_receivers.ReceiverPath`);
across the scan the path resolves through the declarations alone to the class
(`ResolvedProbeOwner`) or the classes of one family (`DeclaredFamilyProbeOwner`).
A receiver no declaration types waits for the type checker's answer
(`UndeclaredProbeOwner`, then `TypeCheckerProbeOwner`); `UnresolvedProbeOwner`
says which step failed.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from .ast_tools import ParsedModule
from .class_index import CompactIndexedClass, module_import_aliases
from .declared_receivers import (
    DeclaredReceiverTyping,
    ReceiverPath,
    ReceiverPathBuilder,
)


@dataclass(frozen=True)
class ReceiverSpan:
    """Where a probe's receiver expression is, as the type checker addresses it."""

    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int

    @classmethod
    def of(cls, file_path: str, node: ast.expr) -> ReceiverSpan:
        return cls(
            file_path,
            node.lineno,
            node.col_offset,
            node.end_lineno or node.lineno,
            node.end_col_offset or node.col_offset,
        )

    @property
    def location(self) -> str:
        """`file:line:column:end_line:end_column`, one-based columns, end inclusive."""

        return (
            f"{self.file_path}:{self.line}:{self.column + 1}:"
            f"{self.end_line}:{self.end_column}"
        )


@dataclass(frozen=True)
class ScannedClasses:
    """The scan's classes by qualified symbol."""

    by_symbol: Mapping[str, CompactIndexedClass]

    def describe_field(self, qualified_name: str, field_name: str) -> str:
        if qualified_name not in self.by_symbol:
            return f"owner `{qualified_name}` is not a scanned class"
        owner = self.by_symbol[qualified_name]
        where = f"`{qualified_name}` ({owner.file_path}:{owner.line})"
        for member in owner.direct_member_declarations:
            if member.name == field_name:
                return (
                    f"owner {where} declares `{field_name}: "
                    f"{member.annotation_expression}`"
                )
        return f"owner {where} declares no `{field_name}` directly"


@dataclass(frozen=True)
class ProbeOwnerContext:
    """The scan's classes and the type checker's owners for unannotated receivers."""

    classes: ScannedClasses
    typed: Mapping[ReceiverSpan, ProbeOwner]
    typing: DeclaredReceiverTyping


class ProbeOwner(ABC):
    """The declared class whose field a foreign absence probe reads, or why none is known."""

    @abstractmethod
    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        raise NotImplementedError

    def type_checker_spans(self) -> tuple[ReceiverSpan, ...]:
        """Receivers this owner needs the type checker for: none once it is known."""

        return ()

    def resolved(self, typing: DeclaredReceiverTyping) -> ProbeOwner:
        """This owner once the scan's declarations have typed it: itself, unless a path."""

        return self

    def class_symbols(self) -> tuple[str, ...]:
        """The scanned class the receiver is known to be, if any."""

        return ()


@dataclass(frozen=True)
class ResolvedProbeOwner(ProbeOwner):
    """The receiver's declared annotation binds to this qualified class."""

    qualified_name: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return context.classes.describe_field(self.qualified_name, field_name)

    def class_symbols(self) -> tuple[str, ...]:
        return (self.qualified_name,)


@dataclass(frozen=True)
class TypeCheckerProbeOwner(ProbeOwner):
    """The type checker infers the receiver as the one scanned class of this name."""

    qualified_name: str
    inferred: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return (
            f"{context.classes.describe_field(self.qualified_name, field_name)} "
            f"(the type checker infers `{self.inferred}`)"
        )

    def class_symbols(self) -> tuple[str, ...]:
        return (self.qualified_name,)


@dataclass(frozen=True)
class UnresolvedProbeOwner(ProbeOwner):
    """No class is known for the receiver; the reason says which step failed."""

    reason: str

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return f"owner unresolved: {self.reason}"


class ReceiverType(ABC):
    """The type checker's answer for one receiver span."""

    @abstractmethod
    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        raise NotImplementedError


@dataclass(frozen=True)
class InferredReceiverType(ReceiverType):
    """The type the checker reports, and the classes it is: one for `X`,
    `type[X]` or `X | None`; none or several for anything else."""

    spelled: str
    classes: tuple[str, ...]

    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        match self.classes:
            case (symbol,) if symbol in classes.by_symbol:
                return TypeCheckerProbeOwner(symbol, self.spelled)
            case (_,):
                return UnresolvedProbeOwner(
                    f"the type checker infers `{self.spelled}`, which is not a scanned class"
                )
        return UnresolvedProbeOwner(
            f"the type checker infers `{self.spelled}`, not one class"
        )


@dataclass(frozen=True)
class FailedReceiverType(ReceiverType):
    """The type checker gave no type: not installed, the build failed, or it typed nothing there."""

    reason: str

    def owner(self, classes: ScannedClasses) -> ProbeOwner:
        return UnresolvedProbeOwner(f"type checker: {self.reason}")


@dataclass(frozen=True)
class DeclaredFamilyProbeOwner(ProbeOwner):
    """The receiver's declarations admit these classes, all of one family."""

    qualified_names: tuple[str, ...]

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return "; ".join(
            context.classes.describe_field(name, field_name)
            for name in self.qualified_names
        )

    def class_symbols(self) -> tuple[str, ...]:
        return self.qualified_names


@dataclass(frozen=True)
class UndeclaredProbeOwner(ProbeOwner):
    """No declaration types the receiver: its owner is whatever the type checker infers."""

    reason: str
    span: ReceiverSpan

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return context.typed[self.span].describe(field_name, context)

    def type_checker_spans(self) -> tuple[ReceiverSpan, ...]:
        return (self.span,)


@dataclass(frozen=True)
class DeclaredPathProbeOwner(ProbeOwner):
    """A receiver as its module declares it; the scan's declarations resolve the path."""

    path: ReceiverPath
    span: ReceiverSpan

    def resolved(self, typing: DeclaredReceiverTyping) -> ProbeOwner:
        declared = self.path.declared(typing)
        match declared.class_symbols():
            case (symbol,):
                return ResolvedProbeOwner(symbol)
            case ():
                return UndeclaredProbeOwner(declared.why_no_class(), self.span)
            case symbols:
                return DeclaredFamilyProbeOwner(symbols)

    def describe(self, field_name: str, context: ProbeOwnerContext) -> str:
        return self.resolved(context.typing).describe(field_name, context)


@dataclass(frozen=True)
class ProbeOwnerResolver:
    """Records each probe receiver as the path its module declares it by."""

    file_path: str
    builder: ReceiverPathBuilder

    @classmethod
    def of(cls, parsed_module: ParsedModule) -> ProbeOwnerResolver:
        return cls(
            parsed_module.file_path,
            ReceiverPathBuilder(
                parsed_module.module_name,
                parsed_module.module,
                module_import_aliases(parsed_module),
            ),
        )

    def owner(self, receiver: ast.expr) -> DeclaredPathProbeOwner:
        return DeclaredPathProbeOwner(
            self.builder.path_of(receiver), ReceiverSpan.of(self.file_path, receiver)
        )


__all__ = (
    "DeclaredFamilyProbeOwner",
    "DeclaredPathProbeOwner",
    "FailedReceiverType",
    "InferredReceiverType",
    "ProbeOwner",
    "ProbeOwnerContext",
    "ProbeOwnerResolver",
    "ReceiverSpan",
    "ReceiverType",
    "ResolvedProbeOwner",
    "ScannedClasses",
    "TypeCheckerProbeOwner",
    "UndeclaredProbeOwner",
    "UnresolvedProbeOwner",
)
