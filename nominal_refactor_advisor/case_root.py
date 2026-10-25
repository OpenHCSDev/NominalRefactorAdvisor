"""NRA's own family root, the one NRA's pyproject declares for CasesFamily.

An enum member was an object carrying its wire value; folded, each member is a
class. ``CaseMeta`` is the type of those classes, so it declares the value they
carry, and boundaries dispatch on it instead of on builtin ``type``.
"""

from __future__ import annotations

from abc import ABC, ABCMeta
from typing import Self


class CaseMeta(ABCMeta):
    """The metaclass of folded case classes: each of its classes carries a value."""

    value: str


class Case(ABC, metaclass=CaseMeta):
    """Root of a folded family; each concrete subclass is one case."""

    @classmethod
    def of(cls, value: object) -> type[Self]:
        """The case ``value`` decodes to, as the enum call it replaces decoded it:
        a case decodes to itself, a value to the case carrying it."""

        for case in cls.__subclasses__():
            if value is case or case.value == value:
                return case
        raise ValueError(f"{value!r} is not a {cls.__name__} value")
