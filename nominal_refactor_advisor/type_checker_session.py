"""One mypy daemon per scan, asked for the types of receivers that declare none.

mypy is an optional extra (`nominal-refactor-advisor[types]`). Without it, or
when the daemon does not start, every receiver gets a `FailedReceiverType`
naming why, so the report says the owner is unresolved and why; nothing is
skipped silently. The daemon is stopped when the session closes.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import ClassVar

from .json_reports import DataclassJsonReport
from .probe_owner import FailedReceiverType, InferredReceiverType, ReceiverSpan, ReceiverType

MYPY_FLAGS = (
    "--strict-equality",
    "--ignore-missing-imports",
    "--python-version",
    "3.11",
)
DAEMON_TIMEOUT_SECONDS = 600
TYPES_EXTRA = "pip install 'nominal-refactor-advisor[types]'"


@dataclass(frozen=True)
class TypeCheckerTiming(DataclassJsonReport):
    """Where a scan's type-checker time went."""

    daemon_start_seconds: float = 0.0
    inspect_seconds: float = 0.0
    inspect_calls: int = 0


class TypeCheckerTimingLedger:
    """Collects the type-checker timing of the scan that opened it."""

    _active: ClassVar[ContextVar[TypeCheckerTimingLedger]]

    def __init__(self) -> None:
        self.timing = TypeCheckerTiming()

    def record(self, timing: TypeCheckerTiming) -> None:
        self.timing = TypeCheckerTiming(
            daemon_start_seconds=round(
                self.timing.daemon_start_seconds + timing.daemon_start_seconds, 3
            ),
            inspect_seconds=round(self.timing.inspect_seconds + timing.inspect_seconds, 3),
            inspect_calls=self.timing.inspect_calls + timing.inspect_calls,
        )

    @classmethod
    def active(cls) -> TypeCheckerTimingLedger:
        return cls._active.get()

    @classmethod
    @contextmanager
    def scope(cls) -> Iterator[TypeCheckerTimingLedger]:
        ledger = cls()
        token = cls._active.set(ledger)
        try:
            yield ledger
        finally:
            cls._active.reset(token)


# Outside a scan's scope the timing is recorded into a ledger nobody reads.
TypeCheckerTimingLedger._active = ContextVar(
    "type_checker_timing_ledger", default=TypeCheckerTimingLedger()
)


class TypeCheckerSession(ABC):
    """The scan's type checker: a running daemon, or the reason there is none."""

    @abstractmethod
    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @staticmethod
    def started(files: Iterable[str]) -> TypeCheckerSession:
        """Start a daemon over these files, or name why none can run."""

        if importlib.util.find_spec("mypy") is None:
            return UnavailableTypeChecker(f"mypy is not installed ({TYPES_EXTRA})")
        scratch = Path(tempfile.mkdtemp(prefix="nra-dmypy-"))
        status_file = scratch / "status.json"
        started = perf_counter()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy.dmypy",
                "--status-file",
                str(status_file),
                "run",
                "--timeout",
                str(DAEMON_TIMEOUT_SECONDS),
                "--export-types",
                "--",
                *MYPY_FLAGS,
                "--cache-dir",
                str(scratch / "cache"),
                *sorted(set(files)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        daemon = DaemonTypeChecker(status_file, scratch, perf_counter() - started)
        # 0 is a clean check and 1 a check with type errors; both leave a daemon.
        if completed.returncode not in (0, 1):
            daemon.stop()
            return UnavailableTypeChecker(
                "the mypy daemon did not start: "
                + (completed.stderr.strip() or completed.stdout.strip())[-500:]
            )
        return daemon

    @staticmethod
    @contextmanager
    def session(files: Iterable[str]) -> Iterator[TypeCheckerSession]:
        session = TypeCheckerSession.started(files)
        try:
            yield session
        finally:
            session.stop()


@dataclass(frozen=True)
class UnavailableTypeChecker(TypeCheckerSession):
    """No daemon: every receiver's type is this failure."""

    reason: str

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        return {span: FailedReceiverType(self.reason) for span in spans}

    def stop(self) -> None:
        """Nothing was started."""


@dataclass(frozen=True)
class DaemonTypeChecker(TypeCheckerSession):
    """A running mypy daemon with exported types."""

    status_file: Path
    scratch: Path
    start_seconds: float

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        client = importlib.import_module("mypy.dmypy.client")
        started = perf_counter()
        answers = {
            span: self.answer(
                client.request(
                    str(self.status_file),
                    "inspect",
                    show="type",
                    location=span.location,
                    verbosity=0,
                    limit=0,
                    include_span=False,
                    include_kind=False,
                    include_object_attrs=False,
                    union_attrs=False,
                    force_reload=False,
                )
            )
            for span in spans
        }
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(
                daemon_start_seconds=self.start_seconds,
                inspect_seconds=perf_counter() - started,
                inspect_calls=len(answers),
            )
        )
        return answers

    @staticmethod
    def answer(response: Mapping[str, object]) -> ReceiverType:
        """The daemon's JSON reply decoded once: a type, or the failure it reports."""

        out = str(response.get("out", "")).strip()
        if response.get("status") == 0 and out:
            return InferredReceiverType(out.splitlines()[0].strip('"'))
        return FailedReceiverType(
            "inspect failed: "
            + (str(response.get("error", "")) or str(response.get("err", "")) or out)
        )

    def stop(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "mypy.dmypy", "--status-file", str(self.status_file), "stop"],
            capture_output=True,
            check=False,
        )
        shutil.rmtree(self.scratch, ignore_errors=True)


__all__ = (
    "DaemonTypeChecker",
    "TypeCheckerSession",
    "TypeCheckerTiming",
    "TypeCheckerTimingLedger",
    "UnavailableTypeChecker",
)
