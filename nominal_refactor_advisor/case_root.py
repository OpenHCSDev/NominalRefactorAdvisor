"""The root every family CasesFamily folds from an enum inherits.

An enum member was an object carrying its wire value; folded, each member is a
class. ``CaseMeta`` is the type of those classes, so it declares the value they
carry, and boundaries dispatch on it instead of on builtin ``type``.
"""

from __future__ import annotations

from abc import ABC, ABCMeta


class CaseMeta(ABCMeta):
    """The metaclass of folded case classes: each of its classes carries a value."""

    value: str


class Case(ABC, metaclass=CaseMeta):
    """Root of a folded family; each concrete subclass is one case."""
