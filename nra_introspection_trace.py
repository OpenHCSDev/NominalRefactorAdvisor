"""Record which project functions introspect which project classes during a real test run.

A function that reads a class's fields through ``dataclasses.fields`` (or
``asdict``, ``astuple``, ``typing.get_type_hints``, ``inspect.signature``)
derives what it does from the declaration; reading two or more classes that
way makes it generic. Static analysis cannot see which classes reach such a
function, because its argument is typed "any dataclass", so the run records
them.

This module needs only the standard library and pytest, so it runs in the
traced project's own environment with nothing but this file on the path:

    PYTHONPATH=<nra checkout> pytest -p nra_introspection_trace \\
        --introspection-trace-out trace.json --introspection-trace-prefix mypackage

``nominal-refactor-advisor --count-member-awareness --introspection-trace trace.json``
reads the file.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
import typing
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import pytest

# Frames inside the introspection machinery itself; the consumer is the first frame past them.
MACHINERY_MODULES = ("dataclasses", "typing", "inspect", "functools", __name__)


@dataclass(frozen=True)
class IntrospectionUse:
    """One function introspecting one class, and how many times the run did it."""

    consumer: str
    introspected: str
    calls: int


@dataclass(frozen=True)
class ClassFieldCount:
    """How many dataclass fields one introspected class declares."""

    introspected: str
    fields: int


@dataclass(frozen=True)
class IntrospectionTrace:
    """Everything one run recorded: the file format, declared once for writer and reader."""

    prefixes: tuple[str, ...]
    uses: tuple[IntrospectionUse, ...]
    field_counts: tuple[ClassFieldCount, ...]

    def encoded(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def decoded(cls, payload: Mapping[str, typing.Any]) -> Self:
        return cls(
            prefixes=tuple(payload["prefixes"]),
            uses=tuple(IntrospectionUse(**use) for use in payload["uses"]),
            field_counts=tuple(ClassFieldCount(**count) for count in payload["field_counts"]),
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.decoded(json.loads(path.read_text(encoding="utf-8")))


class Recording(ABC):
    """What one test run does about introspection: record it, or nothing."""

    @abstractmethod
    def finish(self) -> None:
        raise NotImplementedError

    def pytest_unconfigure(self, config: pytest.Config) -> None:
        self.finish()


class IdleRecording(Recording):
    """The run was not asked for a trace."""

    def finish(self) -> None:
        """Nothing was installed."""


@dataclass
class IntrospectionRecorder(Recording):
    """Wraps the introspection entry points for one run and counts what reaches them."""

    prefixes: tuple[str, ...]
    out: Path
    calls: Counter[tuple[str, str]] = field(default_factory=Counter)
    field_counts: dict[str, int] = field(default_factory=dict)
    originals: dict[tuple[object, str], Callable[..., object]] = field(default_factory=dict)

    def owns(self, module_name: str) -> bool:
        return module_name.startswith(self.prefixes)

    def consumer(self, frame: FrameType | None) -> str | None:
        """The project function past the machinery frames, or None when it is foreign."""
        while frame is not None and frame.f_globals.get("__name__", "").startswith(
            MACHINERY_MODULES
        ):
            frame = frame.f_back
        if frame is None or not self.owns(frame.f_globals.get("__name__", "")):
            return None
        return f"{frame.f_globals['__name__']}.{frame.f_code.co_qualname}"

    def record(self, target: object) -> None:
        introspected = target if isinstance(target, type) else type(target)
        if not self.owns(introspected.__module__):
            return
        consumer = self.consumer(sys._getframe(2))
        if consumer is None:
            return
        name = f"{introspected.__module__}.{introspected.__qualname__}"
        self.calls[(consumer, name)] += 1
        if name not in self.field_counts and dataclasses.is_dataclass(introspected):
            self.field_counts[name] = len(self.originals[(dataclasses, "fields")](introspected))

    def wrap(self, owner: object, name: str) -> None:
        original = getattr(owner, name)
        self.originals[(owner, name)] = original

        def recorded(target: object, *args: object, **kwargs: object) -> object:
            self.record(target)
            return original(target, *args, **kwargs)

        setattr(owner, name, recorded)

    def install(self) -> None:
        for owner, name in (
            (dataclasses, "fields"),
            (dataclasses, "asdict"),
            (dataclasses, "astuple"),
            (typing, "get_type_hints"),
            (inspect, "signature"),
        ):
            self.wrap(owner, name)

    def finish(self) -> None:
        for (owner, name), original in self.originals.items():
            setattr(owner, name, original)
        self.out.write_text(json.dumps(self.trace().encoded(), indent=1), encoding="utf-8")

    def trace(self) -> IntrospectionTrace:
        return IntrospectionTrace(
            prefixes=self.prefixes,
            uses=tuple(
                IntrospectionUse(consumer, introspected, calls)
                for (consumer, introspected), calls in sorted(self.calls.items())
            ),
            field_counts=tuple(
                ClassFieldCount(introspected, fields)
                for introspected, fields in sorted(self.field_counts.items())
            ),
        )


def recording_for(out: Path | None, prefixes: tuple[str, ...]) -> Recording:
    """The pytest options decoded once into what this run records."""
    match out, prefixes:
        case None, _:
            return IdleRecording()
        case Path(), ():
            import pytest

            raise pytest.UsageError(
                "--introspection-trace-out needs at least one --introspection-trace-prefix"
            )
    recorder = IntrospectionRecorder(prefixes, out)
    recorder.install()
    return recorder


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("introspection-trace")
    group.addoption(
        "--introspection-trace-out",
        type=Path,
        help="Write which project functions introspected which project classes here.",
    )
    group.addoption(
        "--introspection-trace-prefix",
        action="append",
        default=[],
        help="A module prefix that is the project (repeatable).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(
        recording_for(
            config.getoption("--introspection-trace-out"),
            tuple(config.getoption("--introspection-trace-prefix")),
        ),
        "nra-introspection-recording",
    )
