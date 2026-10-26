"""The declared-family bases: a marker in the family's own body names it."""

from __future__ import annotations

import ast
from abc import ABC
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Self

from metaclass_registry import AutoRegisterMeta


class DeclaredFamily:
    """A family declared by a marker in its own body, named by its import path."""

    declaration_marker: ClassVar[str]

    @classmethod
    def concrete_families(cls) -> tuple[type[Self], ...]:
        """Every loaded family below this one that declares its own marker."""

        descendants = tuple(
            descendant
            for child in cls.__subclasses__()
            for descendant in (child, *child.concrete_families())
        )
        return tuple(
            family for family in descendants if cls.declaration_marker in vars(family)
        )

    @classmethod
    def import_path(cls) -> str:
        """The qualified path a recipe names this family by."""

        return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class ForeignAdapter(DeclaredFamily, ABC, metaclass=AutoRegisterMeta):
    """A foreign node decoded once into the family that answers questions about it.

    A family root declares ``__registry_key__ = "adapts"`` and so owns exactly
    one registry; each concrete adapter below it is registered under the
    foreign class named by its ``adapts`` attribute.
    """

    declaration_marker = "__registry_key__"
    adapts: ClassVar[type]
    node: ast.AST

    @classmethod
    def adapters(cls) -> tuple[type[Self], ...]:
        """The concrete adapters of this family, in registration order."""

        return tuple(cls.__registry__.values())

    @classmethod
    def member_names(cls) -> frozenset[str]:
        """Qualified names of the foreign classes this family adapts."""

        return frozenset(
            f"{foreign.__module__}.{foreign.__qualname__}"
            for foreign in cls.__registry__
        )

    @classmethod
    def decoded(cls, node: object) -> Self | None:
        """The adapter of this family for ``node``; the family's one type test.

        The most specific registered class on the node's MRO wins, so an
        adapter for a subclass overrides an adapter for its base.
        """

        for foreign in type(node).__mro__:
            adapter = cls.__registry__.get(foreign)
            if adapter is not None:
                return adapter(node)
        return None

    @classmethod
    def within(cls, nodes: Iterable[object]) -> Iterator[Self]:
        """Lazily decode ``nodes`` in order, skipping those this family lacks."""

        return (
            adapter
            for adapter in map(cls.decoded, nodes)
            if adapter is not None
        )
