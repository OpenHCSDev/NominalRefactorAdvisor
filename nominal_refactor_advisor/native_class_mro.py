"""Inert native C3 carriers for already loaded class declarations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Generic

from .class_mro import DeclarationMroType, NativeMroBase
from .native_declarations import ClassNamespaceDeclaration, NativeDeclaration


@dataclass(frozen=True, eq=False)
class NativeClassMroDeclaration(NativeDeclaration, ClassNamespaceDeclaration):
    """Project loaded class ancestry without executing its class-creation hooks."""

    declaration: type

    @property
    def member_binding_names(self) -> frozenset[str]:
        return frozenset(vars(self.declaration))

    def require_generic_origin(self) -> None:
        """Only native typing.Generic subscription is an admitted origin projection."""
        if any(
            "__getitem__" in vars(owner) for owner in type(self.declaration).__mro__
        ):
            raise ValueError("Metaclass subscription has no proved generic origin")
        owner = next(
            (
                owner
                for owner in self.declaration.__mro__
                if "__class_getitem__" in vars(owner)
            ),
            None,
        )
        if owner is not Generic:
            raise ValueError("Custom class subscription has no proved generic origin")

    @property
    @lru_cache(maxsize=None)
    def mro_type(self) -> type:
        terminal = NativeMroBase.for_python_type(self.declaration)
        if terminal is not None:
            return terminal.python_type
        if type(self.declaration).mro is not type.mro:
            raise ValueError(
                f"Native class {self.qualified_name!r} has a custom MRO implementation"
            )
        return DeclarationMroType.from_declaration(
            self,
            tuple(
                NativeClassMroDeclaration(base).mro_type
                for base in self.declaration.__bases__
            ),
        )
