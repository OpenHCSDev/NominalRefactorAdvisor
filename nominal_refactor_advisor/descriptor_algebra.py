"""Reusable descriptor objects for derived structural views."""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import Callable, Generic, TypeVar, cast, overload

ValueT = TypeVar("ValueT")
MemberValueT = TypeVar("MemberValueT")


@dataclass(frozen=True)
class AliasProperty(Generic[ValueT]):
    """Descriptor for properties that are pure aliases of another attribute."""

    source_name: str

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None = None
    ) -> "AliasProperty[ValueT]": ...

    @overload
    def __get__(
        self, instance: object, owner: type[object] | None = None
    ) -> ValueT: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> ValueT | "AliasProperty[ValueT]":
        del owner
        if instance is None:
            return self
        project = cast(Callable[[object], ValueT], attrgetter(self.source_name))
        return project(instance)


@dataclass(frozen=True)
class ConstantProperty(Generic[ValueT]):
    """Descriptor for properties that always return the same immutable value."""

    value: ValueT

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None = None
    ) -> "ConstantProperty[ValueT]": ...

    @overload
    def __get__(
        self, instance: object, owner: type[object] | None = None
    ) -> ValueT: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> ValueT | "ConstantProperty[ValueT]":
        del owner
        if instance is None:
            return self
        return self.value


@dataclass(frozen=True)
class CollectionAttributeProjection(Generic[MemberValueT]):
    """Descriptor projecting one member attribute across an owned collection."""

    collection_name: str
    member_attribute_name: str

    @overload
    def __get__(
        self, instance: None, owner: type[object] | None = None
    ) -> "CollectionAttributeProjection[MemberValueT]": ...

    @overload
    def __get__(
        self, instance: object, owner: type[object] | None = None
    ) -> tuple[MemberValueT, ...]: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> tuple[MemberValueT, ...] | "CollectionAttributeProjection[MemberValueT]":
        del owner
        if instance is None:
            return self
        collection = attrgetter(self.collection_name)
        member_attribute = cast(
            Callable[[object], MemberValueT], attrgetter(self.member_attribute_name)
        )
        return tuple(
            member_attribute(member) for member in collection(instance)
        )
