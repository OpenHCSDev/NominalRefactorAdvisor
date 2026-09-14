"""Bounded memoization for declarations stable during one analysis invocation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import ClassVar, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ScanCache:
    """Own declaration caches until the outer invocation (or worker) ends.

    Per-module AST cleanup cannot invalidate this storage. Outside an explicit
    scope, decorated functions recompute, so interactive callers cannot retain
    identities from a previous implementation. Nested analysis shares its
    invocation; each process-pool worker starts an independent scope.
    """

    _functions: dict[Callable[..., object], Callable[..., object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active: ClassVar[ContextVar[ScanCache | None]] = ContextVar(
        "nra_scan_cache", default=None
    )

    def memoized(self, function: Callable[P, T]) -> Callable[P, T]:
        if function not in self._functions:
            self._functions[function] = lru_cache(maxsize=None)(function)
        # Each key retains only a memoized wrapper of that same callable.
        return cast(Callable[P, T], self._functions[function])

    @classmethod
    def cached(cls, function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def invoke(*args: P.args, **kwargs: P.kwargs) -> T:
            active = cls._active.get()
            implementation = function if active is None else active.memoized(function)
            return implementation(*args, **kwargs)

        return invoke

    @classmethod
    @contextmanager
    def scope(cls) -> Iterator[None]:
        if cls._active.get() is not None:
            yield
            return
        token = cls._active.set(cls())
        try:
            yield
        finally:
            cls._active.reset(token)

    @classmethod
    def initialize_worker(cls) -> None:
        """Begin process-local storage whose lifetime is the analysis pool."""
        cls._active.set(cls())
