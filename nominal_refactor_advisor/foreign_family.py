"""Declared families of foreign classes that are tested together."""

from __future__ import annotations

import ast
from abc import ABCMeta
from collections.abc import Iterable, Iterator
from typing import ClassVar, Self


class ForeignFamilyMeta(ABCMeta):
    """Registers declared members once the ABC machinery exists for the family.

    `__init_subclass__` runs before `ABCMeta` creates the family's registry, so
    a registration made there lands on the parent family's registry instead.
    """

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, object]):
        super().__init__(name, bases, namespace)
        for member in namespace.get("members", ()):
            cls.register(member)


class ForeignFamily(metaclass=ForeignFamilyMeta):
    """Foreign classes that answer one question together, tested once.

    Members declared in a family's own body become virtual subclasses, so ``isinstance`` on the family is the
    single type test and every other site derives from it through ``within``.
    There is deliberately no ``holds`` or ``is_member`` predicate: a predicate
    would turn each if-chain site into a rename; sites are folded later instead.
    """

    members: ClassVar[tuple[type, ...]]

    @classmethod
    def concrete_families(cls) -> tuple[type[Self], ...]:
        """Every loaded family below this one that declares its own members."""

        descendants = tuple(
            descendant
            for child in cls.__subclasses__()
            for descendant in (child, *child.concrete_families())
        )
        return tuple(family for family in descendants if "members" in vars(family))

    @classmethod
    def import_path(cls) -> str:
        """The qualified path a recipe names this family by."""

        return f"{cls.__module__}.{cls.__qualname__}"

    @classmethod
    def member_names(cls) -> frozenset[str]:
        """Qualified names of the classes this family answers for."""

        return frozenset(
            f"{member.__module__}.{member.__qualname__}" for member in cls.members
        )

    @classmethod
    def within(cls, nodes: Iterable[object]) -> Iterator[Self]:
        """Lazily keep the members of this family, in order; the one `isinstance`."""

        return (node for node in nodes if isinstance(node, cls))


class FunctionDefinition(ForeignFamily):
    """A def or async def statement."""

    members = (ast.FunctionDef, ast.AsyncFunctionDef)
