"""Bounded memoization for declarations stable during one analysis invocation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import os
from typing import ClassVar, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ScanCache:
    """Own declaration caches until the outer invocation (or worker) ends.

    Per-module AST cleanup cannot invalidate this storage. Outside an explicit
    scope, the outermost decorated call opens one for its own duration, so
    everything beneath it shares results and interactive callers cannot retain
    identities from a previous implementation. Nested analysis shares its
    invocation; each process-pool worker starts an independent scope.
    """

    _functions: dict[Callable[..., object], Callable[..., object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _deferred_reaps: dict[int, list[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active: ClassVar[ContextVar[ScanCache | None]] = ContextVar(
        "nra_scan_cache", default=None
    )

    @classmethod
    def defer_child_reap(cls, pid: int) -> None:
        """Reap ``pid`` without blocking the critical path when a scope is active.

        Tries a non-blocking ``WNOHANG`` reap first; if the child is still
        tearing down its address space, queues it to be reaped when the active
        ``ScanCache`` scope or forked worker exits. Falls back to blocking
        ``waitpid`` outside an active scope.
        """
        try:
            reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
            if reaped_pid == pid:
                return
        except ChildProcessError:
            return
        active = cls._active.get()
        if active is None:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            return
        active._deferred_reaps.setdefault(os.getpid(), []).append(pid)

    @classmethod
    def reap_deferred_children(cls) -> None:
        """Blocking-reap any children queued by ``os.getpid()`` in the active scope."""
        active = cls._active.get()
        if active is None:
            return
        pids = active._deferred_reaps.pop(os.getpid(), None)
        if not pids:
            return
        for pid in pids:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

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
            if active is None:
                with cls.scope():
                    return invoke(*args, **kwargs)
            return active.memoized(function)(*args, **kwargs)

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
            cls.reap_deferred_children()
            cls._active.reset(token)

    @classmethod
    def active(cls) -> ScanCache | None:
        """The active scan cache in this context, or None outside a scope."""

        return cls._active.get()

    @classmethod
    def initialize_worker(cls) -> None:
        """Begin process-local storage whose lifetime is the analysis pool."""
        cls._active.set(cls())
