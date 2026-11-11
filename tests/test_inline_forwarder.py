"""A one-call forwarder is inlined at every call site and deleted."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPreflightStatus,
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.refactors.function_operations import InlineForwarderOperation


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _inline(root: Path) -> InlineForwarderOperation:
    return InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/identity.py").as_posix(),
            qualname="Actor.require_may_change",
        )
    )


def _identity(root: Path) -> None:
    _write(root, "pkg/__init__.py", "")
    _write(
        root,
        "pkg/identity.py",
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.orders import Todo\n\n\n"
        "class Actor:\n"
        "    name = 'a'\n\n"
        "    def require_may_change(self, todo: Todo) -> None:\n"
        "        todo.require_maintainer(self)\n\n"
        "    def other(self) -> str:\n"
        "        return self.name\n",
    )


def test_a_forwarder_is_inlined_and_its_guarded_import_leaves(tmp_path: Path) -> None:
    _identity(tmp_path)
    _write(
        tmp_path,
        "pkg/orders.py",
        "from pkg.identity import Actor\n\n\n"
        "class Todo:\n"
        "    seen: list = []\n\n"
        "    def require_maintainer(self, actor: Actor) -> None:\n"
        "        Todo.seen.append(actor.name)\n\n\n"
        "class Store:\n"
        "    def __init__(self) -> None:\n"
        "        self.actor = Actor()\n\n"
        "    def change(self, found: list[Todo]) -> None:\n"
        "        self.actor.require_may_change(found[0])\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = RefactorRecipe("inline").with_operation(_inline(tmp_path)).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    simulation.apply()

    identity = (tmp_path / "pkg/identity.py").read_text()
    orders = (tmp_path / "pkg/orders.py").read_text()
    assert "require_may_change" not in identity + orders
    assert "from pkg.orders import Todo" not in identity
    assert "found[0].require_maintainer(self.actor)" in orders
    ran = subprocess.run(
        [sys.executable, "-c",
         "from pkg.orders import Store, Todo; Store().change([Todo()]); assert Todo.seen == ['a']"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert ran.returncode == 0, ran.stderr


@pytest.mark.parametrize(
    ("consumer", "message"),
    (
        ("def bound(actor: Actor):\n    return actor.require_may_change\n", "other than by calling it"),
        ("def used(actor: Actor, todo):\n    return actor.require_may_change(todo)\n", "returns None"),
        (
            "from collections.abc import Callable\n\n\n"
            "def effect(make: Callable[[], Actor], todo):\n    make().require_may_change(todo.pop())\n",
            "reorders the evaluation",
        ),
        ("def untyped(actor, todo):\n    actor.require_may_change(todo)\n", "on a receiver typed"),
        (
            "TABLE = {'change': 'require_may_change'}\n\n\n"
            "def run(actor: Actor, todo, kind: str) -> None:\n    getattr(actor, TABLE[kind])(todo)\n",
            "in a string",
        ),
    ),
)
def test_inlining_is_refused_where_a_call_cannot_inline(
    tmp_path: Path, consumer: str, message: str
) -> None:
    _identity(tmp_path)
    _write(tmp_path, "pkg/orders.py", "class Todo:\n    pass\n")
    _write(tmp_path, "pkg/consumer.py", "from pkg.identity import Actor\n\n\n" + consumer)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    report = _inline(tmp_path).preflight_reports(snapshot)
    assert report and report[0].status is CodemodPreflightStatus.FAILED, report
    assert message in report[0].message


def test_another_class_s_method_of_the_same_name_is_its_own(tmp_path: Path) -> None:
    """Call sites are typed by their receiver: a call on another class that
    declares a method of the same name is that class's method and stays."""
    _identity(tmp_path)
    _write(tmp_path, "pkg/orders.py", "class Todo:\n    def require_maintainer(self, actor) -> None:\n        pass\n")
    _write(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.identity import Actor\nfrom pkg.orders import Todo\n\n\n"
        "class Other:\n    def require_may_change(self, todo: Todo) -> None:\n        pass\n\n\n"
        "def run(actor: Actor, other: Other, todo: Todo) -> None:\n"
        "    actor.require_may_change(todo)\n"
        "    other.require_may_change(todo)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = RefactorRecipe("inline").with_operation(_inline(tmp_path)).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    consumer = simulation.simulation.rewritten_sources[(tmp_path / "pkg/consumer.py").as_posix()]
    assert "    todo.require_maintainer(actor)\n    other.require_may_change(todo)\n" in consumer


def test_an_async_forwarder_with_a_local_import_inlines_and_its_callee_follows(
    tmp_path: Path,
) -> None:
    """agent-comms AttachedChild.send_fenced: an async method that imports its
    callee locally and awaits it. Each awaited call becomes the awaited callee,
    and the callee is imported where the call now is."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/prompt_send.py",
        "SENT: list = []\n\n\nasync def send_fenced_prompt(stream, data) -> None:\n    SENT.append((stream, data))\n",
    )
    _write(
        tmp_path,
        "pkg/child.py",
        "class AttachedChild:\n"
        "    stdin = 'pipe'\n\n"
        "    async def send_fenced(self, data) -> None:\n"
        '        """Keep the raw writer on this input."""\n'
        "        from .prompt_send import send_fenced_prompt\n\n"
        "        await send_fenced_prompt(self.stdin, data)\n",
    )
    _write(
        tmp_path,
        "pkg/native.py",
        "from pkg.child import AttachedChild\n\n\n"
        "class NativeChild:\n"
        "    def __init__(self, proc: AttachedChild) -> None:\n"
        "        self.proc = proc\n\n"
        "    async def send_fenced(self, data) -> None:\n"
        "        await self.proc.send_fenced(data)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/child.py").as_posix(), qualname="AttachedChild.send_fenced",
        )
    )
    simulation = RefactorRecipe("inline").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    simulation.apply()

    child = (tmp_path / "pkg/child.py").read_text()
    native = (tmp_path / "pkg/native.py").read_text()
    assert "send_fenced" not in child and "prompt_send" not in child
    assert "        await send_fenced_prompt(self.proc.stdin, data)\n" in native
    assert "async def send_fenced(self, data)" in native
    ran = subprocess.run(
        [sys.executable, "-c",
         "import asyncio; from pkg.child import AttachedChild; from pkg.native import NativeChild; "
         "from pkg.prompt_send import SENT; asyncio.run(NativeChild(AttachedChild()).send_fenced(b'x')); "
         "assert SENT == [('pipe', b'x')], SENT"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert ran.returncode == 0, ran.stderr


def test_an_ignored_parameter_is_not_inlined_away(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/identity.py",
        "class Actor:\n"
        "    def require_may_change(self, todo) -> None:\n"
        "        self.check()\n\n"
        "    def check(self) -> None:\n"
        "        pass\n",
    )
    _write(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.identity import Actor\n\n\ndef run(actor: Actor, effect):\n    actor.require_may_change(effect())\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    report = _inline(tmp_path).preflight_reports(snapshot)
    assert report[0].status is CodemodPreflightStatus.FAILED
    assert "exactly once" in report[0].message


def test_a_forwarder_that_overrides_its_base_is_refused(tmp_path: Path) -> None:
    """Deleting an override sends every receiver typed as the base to the base's
    answer, though no call this proof types names the subclass (agent-comms
    `SystemRecord.compacted_at` over `ClaudeRecord.compacted_at`)."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/records.py",
        "class Subtype:\n"
        "    def compacted_at(self) -> tuple:\n        return (1,)\n\n\n"
        "class Record:\n"
        "    def compacted_at(self) -> tuple:\n        return ()\n\n\n"
        "class SystemRecord(Record):\n"
        "    def __init__(self, subtype: Subtype) -> None:\n        self.subtype = subtype\n\n"
        "    def compacted_at(self) -> tuple:\n        return self.subtype.compacted_at()\n\n\n"
        "def read(record: Record) -> tuple:\n    return record.compacted_at()\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/records.py").as_posix(),
            qualname="SystemRecord.compacted_at",
        )
    )
    (report,) = operation.preflight_reports(snapshot)
    assert report.status is CodemodPreflightStatus.FAILED, report
    assert "overrides ('Record',)" in report.message


@pytest.mark.parametrize(
    ("consumer", "refused"),
    (
        ("def word(claims):\n    return f'A {claims} report: require_may_change'\n", False),
        ("TABLE = {'change': 'require_may_change'}\nNAMES = ('require_may_change',)\n", False),
        (
            "NAMES = ('require_may_change',)\n\n\nclass Row:\n"
            "    def field(self, name: str):\n        return getattr(self, name)\n",
            False,
        ),
        (
            "NAMES = ('require_may_change',)\n\n\n"
            "def call(actor: Actor, name: str):\n    return getattr(actor, name)\n",
            True,
        ),
        ("NAMES = ('require_may_change',)\n\n\ndef call(cls, name):\n    return vars(cls)[name]\n", True),
    ),
)
def test_only_a_string_that_can_feed_a_lookup_refuses(
    tmp_path: Path, consumer: str, refused: bool
) -> None:
    """A string reaches the method only through a lookup by a non-constant name on
    a family receiver; another class's own lookup, or none at all, leaves it text."""
    _identity(tmp_path)
    _write(tmp_path, "pkg/orders.py", "class Todo:\n    def require_maintainer(self, actor) -> None:\n        pass\n")
    _write(tmp_path, "pkg/consumer.py", "from pkg.identity import Actor\n\n\n" + consumer)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    (report,) = _inline(tmp_path).preflight_reports(snapshot)
    assert (report.status is CodemodPreflightStatus.FAILED) is refused, report


def test_a_property_forwarder_inlines_as_the_chain_it_reads(tmp_path: Path) -> None:
    """agent-comms `TargetField.annotation` returns `self.plan.annotation`: every
    read becomes that chain on the read's receiver, and the property goes."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/fields.py",
        "class Plan:\n    annotation: object = int\n\n\n"
        "class TargetField:\n"
        "    def __init__(self, plan: Plan) -> None:\n        self.plan = plan\n\n"
        "    @property\n    def annotation(self) -> object:\n        return self.plan.annotation\n\n\n"
        "def kind(field: TargetField) -> object:\n    return field.annotation\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/fields.py").as_posix(), qualname="TargetField.annotation"
        )
    )
    simulation = RefactorRecipe("inline").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[(tmp_path / "pkg/fields.py").as_posix()]
    assert "def annotation" not in rewritten
    assert "return field.plan.annotation\n" in rewritten
