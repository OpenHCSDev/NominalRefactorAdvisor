"""Absolute scan-deadline authority shared by CLI and analysis phases."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import Iterator


class ScanDeadlineExceeded(TimeoutError):
    """Raised when a scan crosses its declared absolute wall-clock budget."""

    def __init__(self, deadline: "ScanDeadline") -> None:
        self.budget_seconds = deadline.budget_seconds
        self.elapsed_seconds = deadline.elapsed_seconds
        self.stage = deadline.stage
        super().__init__(
            f"scan deadline exceeded during {self.stage}: "
            f"{self.elapsed_seconds:.3f}s/{self.budget_seconds:.3f}s"
        )


@dataclass
class ScanDeadline:
    """One monotonic deadline with observable phase ownership."""

    budget_seconds: float
    started_at: float
    stage: str = "startup"

    @classmethod
    def start(cls, budget_seconds: float) -> "ScanDeadline":
        return cls(budget_seconds=max(0.0, budget_seconds), started_at=monotonic())

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - self.elapsed_seconds)

    def checkpoint(self, stage: str) -> None:
        self.stage = stage
        if self.elapsed_seconds >= self.budget_seconds:
            raise ScanDeadlineExceeded(self)


_ACTIVE_SCAN_DEADLINE: ContextVar[ScanDeadline | None] = ContextVar(
    "active_scan_deadline",
    default=None,
)


def scan_deadline_checkpoint(stage: str) -> None:
    """Check the active deadline, if any, and publish the current phase."""

    deadline = _ACTIVE_SCAN_DEADLINE.get()
    if deadline is not None:
        deadline.checkpoint(stage)


@contextmanager
def enforce_scan_deadline(deadline: ScanDeadline) -> Iterator[None]:
    """Activate cooperative checks plus a hard main-thread wall timer."""

    token = _ACTIVE_SCAN_DEADLINE.set(deadline)
    alarm_supported = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    previous_handler = None
    if alarm_supported:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def deadline_handler(_signum: int, _frame: object) -> None:
            raise ScanDeadlineExceeded(deadline)

        signal.signal(signal.SIGALRM, deadline_handler)
        signal.setitimer(signal.ITIMER_REAL, deadline.remaining_seconds)
    try:
        yield
    finally:
        if alarm_supported:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
        _ACTIVE_SCAN_DEADLINE.reset(token)
