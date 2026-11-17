"""Python's C3 order of declared classes, read from their source without running them.

A class's order is a function of its bases' orders (C3), so a class lists a
base another of its bases already supplies exactly when dropping it leaves
that order as it is: every lookup, and so the class, is then the same without
it. Composing an owner and dropping a restated base both ask this one question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property

from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.class_mro import (
    DeclarationMroType,
    NativeMroBase,
    RuntimeProofBases,
    namespace_read_native,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput

BASES_READ_AT_CREATION = frozenset(("typing.Protocol", "typing_extensions.Protocol"))
"""Bases whose presence in `__bases__` itself decides what the class is: `Protocol`
makes a class a protocol only when it is a direct base, whatever the order holds."""


class DeclaredC3(ABC):
    """The C3 order of every class `ancestor` can read: scanned, installed or native."""

    @abstractmethod
    def ancestor(self, qualified_name: str) -> CompactIndexedClass | None:
        """The class a base names, or None when nothing NRA reads declares it."""

        raise NotImplementedError

    def declared(self, symbol: str) -> CompactIndexedClass:
        indexed = self.ancestor(symbol)
        if indexed is None:
            raise NeedsInput(
                f"`{symbol}` has no declaration the scan or an installed `.py` source holds",
                NeededInput.PROOF,
            )
        return indexed

    def bases(self, symbol: str) -> tuple[str, ...]:
        """Its bases, each by the symbol it resolves to: a re-export is the class it
        names. Creation's own last bases follow the named ones."""

        declared = self.declared(symbol)
        return tuple(
            qualified_name
            if NativeMroBase.for_qualified_name(qualified_name) or namespace_read_native(qualified_name)
            else self.declared(qualified_name).symbol
            for reference in declared.base_references
            for qualified_name in (".".join(reference.resolved_parts),)
        ) + declared.implicit_base_names or (NativeMroBase.OBJECT.value,)

    @cached_property
    def inert_types(self) -> dict[str, type]:
        """Each class's inert stand-in, once built: Python's own C3 orders them."""

        return {}

    def inert(self, symbol: str) -> type:
        """The inert class standing for `symbol` in an MRO: a native as
        `RuntimeProofBases` stands it in, a declaration by the shared
        `DeclarationMroType` factory over its declared bases."""

        if symbol not in self.inert_types:
            native = RuntimeProofBases().foreign_base(symbol)
            self.inert_types[symbol] = (
                native if native is not None else self.composed(symbol, self.bases(symbol))
            )
        return self.inert_types[symbol]

    def composed(self, symbol: str, bases: tuple[str, ...]) -> type:
        """An inert `symbol` over `bases`; Python refusing their order is a proof
        the class needs."""

        try:
            return DeclarationMroType.from_declaration(
                self.declared(symbol), tuple(self.inert(base) for base in bases)
            )
        except TypeError as refused:
            raise NeedsInput(
                f"`{symbol.rpartition('.')[2]}` has no C3 order with bases "
                + ", ".join(f"`{base.rpartition('.')[2]}`" for base in bases),
                NeededInput.PROOF,
            ) from refused

    @staticmethod
    def entry_symbol(entry: type) -> str:
        """The symbol an inert MRO entry stands for."""

        if isinstance(entry, DeclarationMroType):
            return entry.declaration.symbol
        native = NativeMroBase.for_python_type(entry)
        return entry.__name__ if native is None else native.value

    def linearized(self, symbol: str, bases: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """Python's C3 order of `symbol`, over its declared bases or `bases`."""

        entry = self.inert(symbol) if bases is None else self.composed(symbol, bases)
        return tuple(self.entry_symbol(ancestor) for ancestor in entry.__mro__)

    def restated_bases(
        self, symbol: str, plain: Sequence[bool], leading: tuple[str, ...] = ()
    ) -> tuple[tuple[int, str], ...]:
        """The bases `symbol` lists that another of its bases already supplies, as
        (position among its listed bases, the base supplying it), with `leading`
        first among its bases, as a composition puts its owners.

        `plain` marks each listed base a name or attribute spells: a subscripted
        or computed base is never one. A base is restated when another base's
        order holds it and the class's order without it is exactly its order with
        it; each is tried on the bases the earlier drops left. A base whose
        presence `__bases__` itself reads at creation is never restated.
        """

        listed = self.bases(symbol)
        kept = tuple(base for base in (*leading, *listed) if base != NativeMroBase.OBJECT.value)
        whole = self.linearized(symbol, kept)
        restated: list[tuple[int, str]] = []
        for index, (base, droppable) in enumerate(zip(listed, plain)):
            if not droppable or base in BASES_READ_AT_CREATION:
                continue
            without = tuple(other for other in kept if other != base)
            supplier = next(
                (other for other in without if base in self.linearized(other)[1:]), None
            )
            if supplier is None:
                continue
            try:
                order = self.linearized(symbol, without)
            except NeedsInput:
                continue
            if order == whole:
                kept = without
                restated.append((index, supplier))
        return tuple(restated)


__all__ = ("BASES_READ_AT_CREATION", "DeclaredC3")
