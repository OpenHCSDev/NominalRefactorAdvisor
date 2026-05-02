"""Typed collection normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

ItemT = TypeVar("ItemT")


def sorted_tuple(
    items: Iterable[ItemT],
    *,
    key: Callable[[ItemT], Any] | None = None,
    reverse: bool = False,
) -> tuple[ItemT, ...]:
    ordered = sorted(items, key=key, reverse=reverse)
    return tuple(ordered)
