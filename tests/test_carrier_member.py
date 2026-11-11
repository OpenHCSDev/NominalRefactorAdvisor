"""A text field carrying a family member's declared name becomes the member."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CarrierFieldBecomesMemberOperation,
    CodemodPreflightStatus,
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)

FAMILY = (
    "class Reason:\n"
    "    declared_name = 'reason'\n\n\n"
    "class Timeout(Reason):\n    declared_name = 'timeout'\n\n\n"
    "class Invalid(Reason):\n    declared_name = 'invalid'\n"
)
EVENTS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Error:\n"
    "    text: str\n"
    "    reason_code: str | None = None\n"
)


def _snapshot(tmp_path: Path, consumer: str) -> CodemodSourceSnapshot:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "reasons.py").write_text(FAMILY)
    (package / "events.py").write_text(EVENTS)
    (package / "consumer.py").write_text(consumer)
    return CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))


def _operation(tmp_path: Path) -> CarrierFieldBecomesMemberOperation:
    return CarrierFieldBecomesMemberOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/events.py").as_posix(), qualname="Error.reason_code"
        ),
        family=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/reasons.py").as_posix(), qualname="Reason"
        ),
    )


def test_writers_carry_the_member_and_readers_ask_it(tmp_path: Path) -> None:
    snapshot = _snapshot(
        tmp_path,
        "from pkg import events\n"
        "from pkg.events import Error\n"
        "from pkg.reasons import Invalid, Timeout\n\n\n"
        "def fail() -> Error:\n"
        "    return events.Error(text='x', reason_code=Timeout.declared_name)\n\n\n"
        "def plain() -> Error:\n"
        "    return Error(text='y')\n\n\n"
        "def timed_out(error: Error) -> bool:\n"
        "    return error.reason_code == Timeout.declared_name\n\n\n"
        "def known(error: Error) -> bool:\n"
        "    return error.reason_code in (Timeout.declared_name, Invalid.declared_name)\n\n\n"
        "def label(error: Error) -> str:\n"
        "    return f'failed: {error.reason_code}'\n",
    )
    simulation = RefactorRecipe("carrier").with_operation(_operation(tmp_path)).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    sources = simulation.simulation.rewritten_sources
    events = sources[(tmp_path / "pkg/events.py").as_posix()]
    consumer = sources[(tmp_path / "pkg/consumer.py").as_posix()]
    assert "    reason_code: type[Reason] | None = None\n" in events
    assert "from pkg.reasons import Reason" in events
    assert "events.Error(text='x', reason_code=Timeout)" in consumer
    assert "return error.reason_code is Timeout\n" in consumer
    assert "return error.reason_code in (Timeout, Invalid)\n" in consumer
    assert "f'failed: {error.reason_code.declared_name}'" in consumer


@pytest.mark.parametrize(
    ("consumer", "refusal"),
    (
        (
            "from pkg.events import Error\n\n\ndef fail(code: str) -> Error:\n"
            "    return Error(text='x', reason_code=code)\n",
            "not a Reason member's declared_name",
        ),
        (
            "from pkg.events import Error\n\n\ndef fail() -> Error:\n    return Error('x', None)\n",
            "positionally",
        ),
        (
            "from dataclasses import replace\nfrom pkg.events import Error\n\n\n"
            "def again(error: Error) -> Error:\n    return replace(error, reason_code='timeout')\n",
            "through replace()",
        ),
    ),
)
def test_a_write_that_is_not_a_member_refuses(tmp_path: Path, consumer: str, refusal: str) -> None:
    (report,) = _operation(tmp_path).preflight_reports(_snapshot(tmp_path, consumer))
    assert report.status is CodemodPreflightStatus.FAILED, report
    assert refusal in report.message
