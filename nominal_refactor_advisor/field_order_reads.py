"""Which class code reads a dataclass's fields in their declared order, joined once per scan.

The modules' `FunctionOrderSummary` facts are followed from each class's own methods
through the calls that hand the receiver on, up to `DEPTH` calls, so the verdict that
offers an owner and the apply that composes it refuse a reorder by one proof.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from nominal_refactor_advisor.class_index import (
    ClassMemberAnswersABC,
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
)
from nominal_refactor_advisor.field_order_summaries import FunctionOrderSummary, HandedParameter
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


class DeclaringClass(ClassMemberAnswersABC[tuple[str, ...]]):
    """The class whose `def` a member lookup reaches; none for anything else."""

    def assigned(self, owner: CompactIndexedClass, declaration: object) -> tuple[str, ...]:
        return ()

    def annotated_property(self, owner: CompactIndexedClass, declaration: object) -> tuple[str, ...]:
        return (owner.symbol,)

    def annotated_method(self, owner: CompactIndexedClass, declaration: object) -> tuple[str, ...]:
        return (owner.symbol,)

    def unannotated_callable(self, owner: CompactIndexedClass, declaration: object) -> tuple[str, ...]:
        return (owner.symbol,)

    def missing(self) -> tuple[str, ...]:
        return ()

    def ambiguous(self) -> tuple[str, ...]:
        return ()

    def open(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class FieldOrderReads:
    """The scan's ordered field reads, joined across modules once."""

    projections: Sequence[CompactModuleClassProjection]
    class_index: CompactClassFamilyIndex
    DEPTH: ClassVar[int] = 4

    @cached_property
    def by_symbol(self) -> Mapping[str, FunctionOrderSummary]:
        return {
            summary.symbol: summary
            for projection in self.projections
            for summary in projection.field_order_summaries
        }

    @cached_property
    def by_class(self) -> Mapping[str, tuple[FunctionOrderSummary, ...]]:
        grouped: dict[str, list[FunctionOrderSummary]] = {}
        for summary in self.by_symbol.values():
            if summary.class_symbol is not None:
                grouped.setdefault(summary.class_symbol, []).append(summary)
        return {symbol: tuple(found) for symbol, found in grouped.items()}

    @cached_property
    def file_by_module(self) -> Mapping[str, str]:
        return {projection.module_name: projection.file_path for projection in self.projections}

    def read_of(self, reordered: Iterable[str]) -> str | None:
        """Where class code reads the order of a class in `reordered`, or of a
        descendant, with the move's consequence; none when nothing reads it."""

        family = dict.fromkeys(
            member
            for symbol in reordered
            for member in (symbol, *self.class_index.descendant_symbols(symbol))
        )
        seen: set[str] = set()
        for member in family:
            for symbol in (member, *self.class_index.ancestor_symbols(member)):
                if symbol in seen or self.class_index.class_for(symbol) is None:
                    continue
                seen.add(symbol)
                read = self.first_read(symbol)
                if read is not None:
                    reordered_class = self.class_index.class_for(member)
                    name = member if reordered_class is None else reordered_class.qualname
                    return (
                        f"{read}: the move reorders the fields of `{name}`, "
                        "which changes what it builds"
                    )
        return None

    def require_unread(self, reordered: Iterable[str]) -> None:
        read = self.read_of(reordered)
        if read is not None:
            raise NeedsInput(read, NeededInput.SOURCE_CHANGE)

    def first_read(self, symbol: str) -> str | None:
        """Where a function the class `symbol` declares reads its receiver's field order."""

        indexed_class = self.class_index.class_for(symbol)
        qualname = symbol if indexed_class is None else indexed_class.qualname
        for summary in self.by_class.get(symbol, ()):
            receiver = summary.receiver
            if receiver is None:
                continue
            path = self.reads(summary, receiver, 0, frozenset())
            if path is not None:
                module = symbol.rsplit(".", 1)[0] if indexed_class is None else indexed_class.module_name
                where = self.file_by_module.get(module, module)
                return (
                    f"`{qualname}.{summary.name}` ({where}:{summary.line}) reads its class's "
                    "dataclass fields in order through " + " -> ".join(f"`{step}`" for step in path)
                )
        return None

    def reads(
        self,
        summary: FunctionOrderSummary,
        parameter: str,
        depth: int,
        visited: frozenset[tuple[str, str]],
    ) -> tuple[str, ...] | None:
        """The calls through which `parameter` reaches an ordered read, ending at the read."""

        key = (summary.symbol, parameter)
        if depth > self.DEPTH or key in visited:
            return None
        visited = visited | {key}
        for read_parameter, read in summary.reads:
            if read_parameter == parameter:
                return (read,)
        for handed in summary.handed:
            if handed.parameter != parameter:
                continue
            callee = self.callee(summary, handed)
            if callee is None:
                continue
            target, label, bound_method = callee
            bound = target.bound(handed, bound_method)
            if bound is None:
                continue
            rest = self.reads(target, bound, depth + 1, visited)
            if rest is not None:
                return (label, *rest)
        return None

    def callee(
        self, summary: FunctionOrderSummary, handed: HandedParameter
    ) -> tuple[FunctionOrderSummary, str, bool] | None:
        """The scanned function a call reaches: a method of the caller's own
        receiver, a class's method by its dotted name, or a top-level function."""

        if handed.receiver_method is not None and summary.class_symbol is not None:
            return self.method(summary.class_symbol, handed.receiver_method)
        if handed.callee is None:
            return None
        owner, _, name = handed.callee.rpartition(".")
        if self.class_index.class_for(owner) is not None:
            return self.method(owner, name)
        target = self.by_symbol.get(handed.callee)
        return None if target is None or target.class_symbol is not None else (target, name, False)

    def method(self, owner_symbol: str, name: str) -> tuple[FunctionOrderSummary, str, bool] | None:
        for symbol in self.class_index.declared_member(owner_symbol, name).answer(DeclaringClass()):
            target = self.by_symbol.get(f"{symbol}.{name}")
            if target is not None:
                indexed_class = self.class_index.class_for(symbol)
                qualname = symbol if indexed_class is None else indexed_class.qualname
                return target, f"{qualname}.{name}", True
        return None


__all__ = ("DeclaringClass", "FieldOrderReads")
