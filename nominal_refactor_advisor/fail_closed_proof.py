"""Proofs that hold only when none of their obligations fails."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import ClassVar

from nominal_refactor_advisor.fielded_enum import FieldedStrEnum


class ProofObligation(FieldedStrEnum):
    """One thing a fail-closed proof must show.

    A member is written `NAME = "key", "what is wrong when it fails", test`.  The test
    takes the proof and answers whether the obligation fails.
    """

    explanation: str
    _predicate: Callable[[FailClosedProof], bool]

    def is_violated_by(self, proof: FailClosedProof) -> bool:
        return self._predicate(proof)


class FailClosedProof:
    """A proof that holds only when none of its obligations fails.

    A proof class names the obligations it adds with `obligations=` and answers for
    those of every proof class it inherits, its own first.
    """

    _obligations: ClassVar[tuple[ProofObligation, ...]] = ()

    def __init_subclass__(
        cls, obligations: type[ProofObligation] | None = None, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        inherited = dict.fromkeys(
            obligation
            for base in cls.__bases__
            if issubclass(base, FailClosedProof)
            for obligation in base._obligations
        )
        cls._obligations = (*(() if obligations is None else obligations), *inherited)

    @cached_property
    def violations(self) -> tuple[ProofObligation, ...]:
        """Every obligation that fails, in the order the proof answers for them."""

        return tuple(
            obligation
            for obligation in type(self)._obligations
            if obligation.is_violated_by(self)
        )

    @property
    def is_proven(self) -> bool:
        """Whether no obligation fails.  It stops at the first one that does."""

        return not any(
            obligation.is_violated_by(self) for obligation in type(self)._obligations
        )

    @property
    def rejection_reason(self) -> str:
        return "; ".join(violation.explanation for violation in self.violations)
