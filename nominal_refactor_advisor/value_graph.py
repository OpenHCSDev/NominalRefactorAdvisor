"""Declaration-derived value comparison for immutable dataclass DAGs."""

from abc import ABC
from collections.abc import Callable
from dataclasses import Field, fields
from functools import partial
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


class DataclassGraphValue(ABC):
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
