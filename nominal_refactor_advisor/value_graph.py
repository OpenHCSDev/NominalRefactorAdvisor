"""Declaration-owned graph traversal and immutable value comparison."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterator
from dataclasses import Field, fields
from functools import cached_property, partial
from typing import Any


def _children(
    value: object,
    include_field: Callable[[Field[Any]], bool],
    inherits_operation: Callable[[type], bool],
    root: object,
) -> tuple[object, ...] | None:
    if type(value) is tuple:
        return value
    if isinstance(value, DataclassGraphValue) and (
        value is root or inherits_operation(type(value))
    ):
        return tuple(
            (
                getattr(value, declaration.name)
                for declaration in fields(value)
                if include_field(declaration)
            )
        )
    return None


class _HashValue:
    """Feed an already derived child hash into native tuple hashing."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __hash__(self) -> int:
        return self.value


class StoredDataclassState:
    """Opt in only when declared field getters expose ordinary stored values.

    Graph traversal alone does not establish this transport contract: descriptor
    projections may differ from raw storage. Nonparticipating graph nodes keep
    their existing native pickle behavior.
    """

    def __getstate__(self) -> dict[str, object]:
        return {
            declaration.name: getattr(self, declaration.name)
            for declaration in fields(self)
        }


class CachedDataclassRepresentation(StoredDataclassState):
    """Opt-in representation for acyclic, transitively immutable stored records.

    Declare participants with ``frozen=True, repr=False``. Field values must have
    stable representations throughout the record lifetime. Declared fields and
    their repr flags remain the authority; the derived text is not transported.
    """

    @cached_property
    def _representation(self) -> str:
        members = ", ".join(
            f"{declaration.name}={getattr(self, declaration.name)!r}"
            for declaration in fields(self)
            if declaration.repr
        )
        return f"{type(self).__qualname__}({members})"

    def __repr__(self) -> str:
        return self._representation


class DataclassGraphNode(ABC):
    """Declared record ownership, independent of comparison and hash behavior.

    Only opted-in dataclass nodes and exact tuple containers are traversed.
    Runtime objects, classes and nonparticipating records remain opaque.
    Shared identities and cycles are visited once, without invoking equality.
    """

    @property
    def graph_children(self) -> tuple[object, ...]:
        """Declared edges; dataclass fields supply the default projection."""
        return tuple(getattr(self, declaration.name) for declaration in fields(self))

    def graph_nodes(self) -> Iterator[DataclassGraphNode]:
        pending: list[object] = [self]
        seen: set[int] = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if type(value) is tuple:
                pending.extend(reversed(value))
            elif issubclass(type(value), DataclassGraphNode):
                yield value
                pending.extend(reversed(value.graph_children))


class DataclassGraphValue(DataclassGraphNode):
    """Opt-in field equality and hashing with traversal-local DAG memoization.

    Participants are immutable dataclasses declared with eq=False. Inherited
    fields and their compare/hash flags remain the authority. Exact tuples
    are traversed structurally; other values keep native leaf semantics.
    Custom comparison/hash overrides form opaque boundaries. Cycles reached
    during traversal raise ValueError; cyclic values are outside this contract.
    No derived hashes or traversal state are retained on values.
    """

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        pending = [(self, other, False)]
        complete: dict[tuple[int, int], tuple[object, object]] = {}
        active: set[tuple[int, int]] = set()
        children_for = partial(
            _children,
            include_field=lambda declaration: declaration.compare,
            inherits_operation=lambda cls: cls.__eq__ is DataclassGraphValue.__eq__,
        )
        while pending:
            left, right, finishing = pending.pop()
            pair = (id(left), id(right))
            if finishing:
                active.remove(pair)
                complete[pair] = (left, right)
                continue
            if left is right or pair in complete:
                continue
            if pair in active:
                raise ValueError("DataclassGraphValue requires an acyclic value graph")
            children = children_for(left, root=self)
            if children is None or type(left) is not type(right):
                if not left == right:
                    return False
                complete[pair] = (left, right)
                continue
            other_children = children_for(right, root=other)
            assert other_children is not None
            if len(children) != len(other_children):
                return False
            active.add(pair)
            pending.append((left, right, True))
            pending.extend(
                (
                    (left_child, right_child, False)
                    for left_child, right_child in reversed(
                        tuple(zip(children, other_children))
                    )
                )
            )
        return True

    def __hash__(self) -> int:
        pending: list[tuple[object, tuple[object, ...] | None]] = [(self, None)]
        derived: dict[int, tuple[object, int]] = {}
        active: set[int] = set()
        children_for = partial(
            _children,
            include_field=lambda declaration: (
                declaration.compare if declaration.hash is None else declaration.hash
            ),
            inherits_operation=lambda cls: cls.__hash__ is DataclassGraphValue.__hash__,
            root=self,
        )
        while pending:
            value, finishing_children = pending.pop()
            identity = id(value)
            if finishing_children is not None:
                derived[identity] = (
                    value,
                    hash(
                        tuple(
                            (
                                _HashValue(derived[id(child)][1])
                                for child in finishing_children
                            )
                        )
                    ),
                )
                active.remove(identity)
                continue
            if identity in derived:
                continue
            if identity in active:
                raise ValueError("DataclassGraphValue requires an acyclic value graph")
            children = children_for(value)
            if children is None:
                derived[identity] = (value, hash(value))
                continue
            active.add(identity)
            pending.append((value, children))
            pending.extend(((child, None) for child in reversed(children)))
        return derived[id(self)][1]
