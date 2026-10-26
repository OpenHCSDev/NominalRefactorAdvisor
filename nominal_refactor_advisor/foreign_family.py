"""Declared families of foreign classes that are tested together."""

from __future__ import annotations

import ast
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator
from typing import ClassVar, Self

from .ast_projection import AstExpressionProjection
from .declared_family import DeclaredFamily, ForeignAdapter


class ForeignFamilyMeta(ABCMeta):
    """Registers declared members once the ABC machinery exists for the family.

    `__init_subclass__` runs before `ABCMeta` creates the family's registry, so
    a registration made there lands on the parent family's registry instead.
    """

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, object]):
        super().__init__(name, bases, namespace)
        for member in namespace.get("members", ()):
            cls.register(member)


class ForeignFamily(DeclaredFamily, metaclass=ForeignFamilyMeta):
    """Foreign classes that answer one question together, tested once.

    Members declared in a family's own body become virtual subclasses, so ``isinstance`` on the family is the
    single type test and every other site derives from it through ``within``.
    There is deliberately no ``holds`` or ``is_member`` predicate: a predicate
    would turn each if-chain site into a rename; sites are folded later instead.
    """

    declaration_marker = "members"
    members: ClassVar[tuple[type, ...]]

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
