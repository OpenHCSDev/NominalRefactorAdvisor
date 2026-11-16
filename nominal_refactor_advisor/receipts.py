"""Receipts: the facts behind a score, each with a fixed weight anyone can re-add.

A receipt kind carries a log-odds weight per unit and says whether it is a
guard. A guard weighs nothing and stops the move it is attached to until the input
its kind names is supplied. A
receipt is one fact of a kind: how many units of it, what it says, and where.
Class-overlap verdicts (#70) and owner candidates (#67) both score with
receipts, each over its own family of kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from nominal_refactor_advisor.json_reports import SemanticRecord, json_report_property
from nominal_refactor_advisor.models import SourceLocation
from nominal_refactor_advisor.needed_input import NeededInput


class ReceiptKind(StrEnum):
    """One kind of evidence, with its log-odds weight per unit, whether it guards,
    and, for a guard, the kind of input that clears it."""

    weight: float
    guards: bool
    needs: NeededInput

    def __new__(
        cls,
        value: str,
        weight: float,
        guards: bool,
        needs: NeededInput = NeededInput.UNTYPED,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.weight = weight
        member.guards = guards
        member.needs = needs
        return member


@dataclass(frozen=True)
class Receipt(SemanticRecord):
    """One fact behind a score: its kind, how many units, what it says, where."""

    kind: ReceiptKind
    detail: str
    evidence: tuple[SourceLocation, ...]
    count: int = 1

    @json_report_property()
    def weight(self) -> float:
        return self.kind.weight * self.count


@dataclass(frozen=True)
class KeywordCuredReceipt(Receipt):
    """A guard that building these classes by keyword clears: they are constructed
    by position, so a move that reorders or adds their fields would rebind those
    calls. Keywording every construction first leaves nothing to rebind."""

    constructed: tuple[str, ...] = ()


class ScoredByReceipts:
    """A verdict whose score is the sum of its receipts and whose guards name what it needs."""

    receipts: tuple[Receipt, ...]

    @json_report_property()
    def log_odds(self) -> float:
        return sum(receipt.weight for receipt in self.receipts)

    @property
    def refusals(self) -> tuple[Receipt, ...]:
        """The guards that fired: each names an input the move needs."""

        return tuple(receipt for receipt in self.receipts if receipt.kind.guards)

    @json_report_property()
    def permitted(self) -> bool:
        return not self.refusals

    @property
    def keyword_cure(self) -> tuple[str, ...]:
        """The classes to build by keyword first, when that clears every guard that
        fired; empty when nothing fired or some guard has no such cure."""

        refusals = self.refusals
        cured = [receipt for receipt in refusals if isinstance(receipt, KeywordCuredReceipt)]
        if not refusals or len(cured) != len(refusals):
            return ()
        return tuple(dict.fromkeys(symbol for receipt in cured for symbol in receipt.constructed))


__all__ = ("KeywordCuredReceipt", "Receipt", "ReceiptKind", "ScoredByReceipts")
