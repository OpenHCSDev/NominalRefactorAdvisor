"""Receipts: the facts behind a score, each with a fixed weight anyone can re-add.

A receipt kind carries a log-odds weight per unit and says whether it is a
guard. A guard weighs nothing and refuses the move it is attached to. A
receipt is one fact of a kind: how many units of it, what it says, and where.
Class-overlap verdicts (#70) and owner candidates (#67) both score with
receipts, each over its own family of kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from .json_reports import SemanticRecord, json_report_property
from .models import SourceLocation


class ReceiptKind(StrEnum):
    """One kind of evidence, with its log-odds weight per unit and whether it guards."""

    weight: float
    guards: bool

    def __new__(cls, value: str, weight: float, guards: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.weight = weight
        member.guards = guards
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


class ScoredByReceipts:
    """A verdict whose score is the sum of its receipts and whose guards refuse it."""

    receipts: tuple[Receipt, ...]

    @json_report_property()
    def log_odds(self) -> float:
        return sum(receipt.weight for receipt in self.receipts)

    @property
    def refusals(self) -> tuple[Receipt, ...]:
        """The guards that fired: any one refuses."""

        return tuple(receipt for receipt in self.receipts if receipt.kind.guards)

    @json_report_property()
    def permitted(self) -> bool:
        return not self.refusals


__all__ = ("Receipt", "ReceiptKind", "ScoredByReceipts")
