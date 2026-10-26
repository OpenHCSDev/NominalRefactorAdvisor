"""Declared families of foreign classes that are tested together."""

from __future__ import annotations

import ast
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar, Self

from metaclass_registry import AutoRegisterMeta

from .ast_projection import AstExpressionProjection


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


@dataclass(frozen=True)
class ForeignAdapter(ABC, metaclass=AutoRegisterMeta):
    """A foreign node decoded once into the family that answers questions about it.

    A family root declares ``__registry_key__ = "adapts"`` and so owns exactly
    one registry; each concrete adapter below it is registered under the
    foreign class named by its ``adapts`` attribute.
    """

    adapts: ClassVar[type]
    node: ast.AST

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


class AssignmentStatement(ForeignAdapter):
    """An assignment statement: what it binds, to which value, under which annotation."""

    __registry_key__ = "adapts"

    @property
    @abstractmethod
    def targets(self) -> tuple[ast.expr, ...]:
        """Every expression the statement binds."""

    @property
    def value(self) -> ast.expr | None:
        """The assigned expression, absent for a bare annotation."""

        return self.node.value

    @property
    def annotation(self) -> ast.expr | None:
        """The declared annotation, absent unless the statement carries one."""

        return None

    @property
    def single_name(self) -> str | None:
        """The identifier when exactly one target is a bare name."""

        match self.targets:
            case (target,):
                return AstExpressionProjection.identifier(target)
            case _:
                return None


class AssignStatement(AssignmentStatement):
    """``a = b = value``."""

    adapts = ast.Assign

    @property
    def targets(self) -> tuple[ast.expr, ...]:
        return tuple(self.node.targets)


class AnnAssignStatement(AssignmentStatement):
    """``target: annotation`` with or without a value."""

    adapts = ast.AnnAssign

    @property
    def targets(self) -> tuple[ast.expr, ...]:
        return (self.node.target,)

    @property
    def annotation(self) -> ast.expr | None:
        return self.node.annotation
