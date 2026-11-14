"""String enums whose members carry declared fields."""

from __future__ import annotations

import inspect
from enum import StrEnum
from typing import Self


class FieldedStrEnum(StrEnum):
    """A string enum whose members carry the fields its body annotates.

    A member is written `NAME = "value", first, second`: the value, then one entry per
    annotated name, in the order the class (and its enum parents) annotate them.  A
    member with too few or too many entries fails when the class is created.
    """

    def __new__(cls, value: str, *fields: object) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        for name, field in zip(cls.member_field_names(), fields, strict=True):
            setattr(member, name, field)
        return member

    @classmethod
    def member_field_names(cls) -> tuple[str, ...]:
        """The names a member's extra values are stored under, parents' first."""

        return tuple(
            name
            for owner in reversed(cls.__mro__)
            if issubclass(owner, FieldedStrEnum)
            for name, annotation in inspect.get_annotations(owner).items()
            if "ClassVar" not in str(annotation).partition("[")[0]
        )
