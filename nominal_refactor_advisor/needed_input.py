"""What a refactor still needs before NRA can apply it, and who supplies it.

A check that stops a change is asking for something. Naming which kind of
input it asks for, on every synthesis record, lets one scan list everything
the remaining findings need and lets each kind be supplied in one pass, the
way owner picks already are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Self


class NeededInput(StrEnum):
    """One kind of missing input, with who supplies it."""

    supplied_by: str

    OWNER_PICK = "owner_pick", "a picker, in --codemod-owner-picks"
    SOURCE_CHANGE = "source_change", "the target code, changed before the move"
    PROOF = "proof", "NRA: a proof it does not make yet"
    OPERATION = "operation", "NRA: an operation nobody has written yet"
    UNTYPED = "untyped", "nobody yet: the reason is free text, so type its check"

    def __new__(cls, value: str, supplied_by: str) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.supplied_by = supplied_by
        return member

    @classmethod
    def distinct(cls, needs: Iterable[NeededInput]) -> tuple[NeededInput, ...]:
        """Each kind once, in declaration order; nothing typed reads as untyped."""

        present = frozenset(needs)
        return tuple(member for member in cls if member in present) or (cls.UNTYPED,)


class NeedsInput(ValueError):
    """A check that stops a change, naming the kinds of input it needs."""

    def __init__(self, message: str, *needs: NeededInput) -> None:
        super().__init__(message)
        self.needs = NeededInput.distinct(needs)

    def __reduce__(self) -> tuple[type[NeedsInput], tuple[object, ...]]:
        return (type(self), (self.args[0], *self.needs))

    @classmethod
    def every(cls, *checks: Callable[[], object]) -> None:
        """Run each independent check and refuse once, naming every check that
        refused: one scan then shows all a step still needs, not its first."""

        refused: list[ValueError] = []
        for check in checks:
            try:
                check()
            except ValueError as error:
                refused.append(error)
        if len(refused) == 1:
            raise refused[0]
        if refused:
            raise cls(
                "; ".join(str(error) for error in refused),
                *(need for error in refused for need in cls.needs_of(error)),
            )

    @staticmethod
    def needs_of(error: BaseException) -> tuple[NeededInput, ...]:
        """The kinds every `NeedsInput` along `error`'s cause chain names.

        The chain is foreign until decoded: any exception can sit in it, so this
        is the one place that asks which ones name a need.
        """

        found: list[NeededInput] = []
        visited: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, NeedsInput):
                found.extend(current.needs)
            current = current.__cause__ or current.__context__
        return NeededInput.distinct(found)
