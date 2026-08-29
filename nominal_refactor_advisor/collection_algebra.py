"""Typed collection normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

ItemT = TypeVar("ItemT")
IdentityHandleT = TypeVar("IdentityHandleT")
IdentityDeclarationT = TypeVar("IdentityDeclarationT")
IdentityValueT = TypeVar("IdentityValueT")


class IdentityHandleCollisionError(ValueError):
    """One compact handle resolved to two unequal declarations."""

    def __init__(
        self,
        handle: object,
        existing_declaration: object,
        colliding_declaration: object,
    ) -> None:
        self.handle = handle
        self.existing_declaration = existing_declaration
        self.colliding_declaration = colliding_declaration
        super().__init__(
            f"Identity handle {handle!r} resolves to unequal declarations: "
            f"{existing_declaration!r} and {colliding_declaration!r}"
        )


class UniqueIdentityIndexAuthority(
    Generic[IdentityHandleT, IdentityDeclarationT, IdentityValueT]
):
    """Build a handle index while proving every handle has one declaration."""

    _declarations_by_handle: dict[IdentityHandleT, IdentityDeclarationT]
    _values_by_handle: dict[IdentityHandleT, IdentityValueT]

    def __init__(self) -> None:
        self._declarations_by_handle = {}
        self._values_by_handle = {}

    def add(
        self,
        handle: IdentityHandleT,
        declaration: IdentityDeclarationT,
        value: IdentityValueT,
    ) -> None:
        if handle not in self._declarations_by_handle:
            self._declarations_by_handle[handle] = declaration
            self._values_by_handle[handle] = value
            return
        existing_declaration = self._declarations_by_handle[handle]
        if existing_declaration != declaration:
            raise IdentityHandleCollisionError(
                handle,
                existing_declaration,
                declaration,
            )

    def values_by_handle(self) -> dict[IdentityHandleT, IdentityValueT]:
        return dict(self._values_by_handle)

    @classmethod
    def declarations_by_handle(
        cls,
        declarations: Iterable[IdentityDeclarationT],
        handle_for: Callable[[IdentityDeclarationT], IdentityHandleT],
    ) -> dict[IdentityHandleT, IdentityDeclarationT]:
        index = cls[IdentityHandleT, IdentityDeclarationT, IdentityDeclarationT]()
        for declaration in declarations:
            index.add(handle_for(declaration), declaration, declaration)
        return index.values_by_handle()


def sorted_tuple(
    items: Iterable[ItemT],
    *,
    key: Callable[[ItemT], Any] | None = None,
    reverse: bool = False,
) -> tuple[ItemT, ...]:
    ordered = sorted(items, key=key, reverse=reverse)
    return tuple(ordered)
