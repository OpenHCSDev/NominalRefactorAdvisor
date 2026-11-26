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
        ("NAMES = ('require_may_change',)\n\n\ndef call(name):\n    return vars(Actor)[name]\n", True),
        (
            "class Other:\n    def run(self):\n        pass\n\n\nNAMES = ('require_may_change',)\n\n\n"
            "def patch(fake):\n    for owner, method in ((Other, 'run'),):\n        setattr(owner, method, fake)\n",
            False,
        ),
        (
            "NAMES = ('require_may_change',)\n\n\n"
            "def patch(fake):\n    for owner, method in ((Actor, 'require_may_change'),):\n        setattr(owner, method, fake)\n",
            True,
        ),
        ("NAMES = ('require_may_change',)\n\n\ndef call(actor: Actor, name):\n    return type(actor).__dict__[name]\n", True),
        ("NAMES = ('require_may_change',)\n\n\ndef call(actor: Actor, name):\n    return vars(actor)[name]\n", False),
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


def test_a_cooperative_mi_mixin_forwarder_overriding_a_sibling_base_is_refused(
    tmp_path: Path,
) -> None:
    """agent-comms `AssignmentDecision` placed before `AssignmentState` in
    `BoundAssignment(AssignmentDecision, AssignmentState)`: deleting `mode`
    un-implements `AssignmentState.mode` (`@abstractmethod`) and deleting
    `verdict` falls through to `UnengagedDefaults.verdict` (`None`) on every
    receiver typed as `AssignmentState`."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/assignments.py",
        "from abc import ABC, abstractmethod\n\n\n"
        "class Engagement:\n"
        "    mode = 'full'\n"
        "    verdict = 'engage'\n\n\n"
        "class UnengagedDefaults:\n"
        "    @property\n    def verdict(self) -> str | None:\n        return None\n\n\n"
        "class AssignmentState(UnengagedDefaults, ABC):\n"
        "    @property\n    @abstractmethod\n    def mode(self) -> str:\n        raise NotImplementedError\n\n\n"
        "class AssignmentDecision:\n"
        "    decision: Engagement\n\n"
        "    @property\n    def mode(self) -> str:\n        return self.decision.mode\n\n"
        "    @property\n    def verdict(self) -> str | None:\n        return self.decision.verdict\n\n\n"
        "class BoundAssignment(AssignmentDecision, AssignmentState):\n"
        "    def __init__(self, decision: Engagement) -> None:\n        self.decision = decision\n\n\n"
        "def read(state: AssignmentState) -> tuple[str, str | None]:\n"
        "    return state.mode, state.verdict\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    for member, overridden in (("mode", "AssignmentState"), ("verdict", "UnengagedDefaults")):
        operation = InlineForwarderOperation(
            target=SourceRewriteTarget(
                file_path=(tmp_path / "pkg/assignments.py").as_posix(),
                qualname=f"AssignmentDecision.{member}",
            )
        )
        (report,) = operation.preflight_reports(snapshot)
        assert report.status is CodemodPreflightStatus.FAILED, report
        assert f"overrides ({overridden!r},)" in report.message


@pytest.mark.parametrize(
    ("extra_classes", "consumer_sig", "consumer_body", "expected_fragment"),
    (
        (
            "class Model:\n    pass\n\n\n",
            "model: Model",
            "active = model\n    return active.running_router()\n",
            "can hold a `pkg.models.DraftedModel` instance",
        ),
        (
            "class Model:\n    pass\n\n\nclass Mixin:\n    pass\n\n\n"
            "class Combined(Mixin, DraftedModel):\n    pass\n\n\n",
            "mixin: Mixin",
            "return mixin.running_router()\n",
            "can hold a `pkg.models.DraftedModel` instance",
        ),
        (
            "from typing import Protocol\n\n\nclass Model:\n    pass\n\n\n"
            "class SupportsRouter(Protocol):\n    def running_router(self) -> int: ...\n\n\n",
            "proto: SupportsRouter",
            "return proto.running_router()\n",
            "can hold a `pkg.models.DraftedModel` instance",
        ),
        (
            "class Model:\n    pass\n\n\n",
            "value: object",
            "return value.running_router()\n",
            "can hold a `pkg.models.DraftedModel` instance",
        ),
    ),
)
def test_supertype_mi_protocol_and_object_receivers_refuse_inlining(
    tmp_path: Path,
    extra_classes: str,
    consumer_sig: str,
    consumer_body: str,
    expected_fragment: str,
) -> None:
    """personal-model `DraftedModel.running_router`: a receiver typed as an
    un-narrowed ancestor (`Model`), an MI sibling with a common subclass, a
    structural `Protocol`, or `object` can hold a `DraftedModel` at runtime
    and must not be silently skipped."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/models.py",
        "class Reader:\n    def running_router(self) -> int:\n        return 7\n\n\n"
        + extra_classes
        + "class DraftedModel(Model):\n"
        "    def __init__(self, reader: Reader) -> None:\n        self.reader = reader\n\n"
        "    def running_router(self) -> int:\n        return self.reader.running_router()\n",
    )
    # Place `Combined` after `DraftedModel` when the fixture declares it.
    models_path = tmp_path / "pkg/models.py"
    text = models_path.read_text()
    if "class Combined(Mixin, DraftedModel):" in text:
        text = text.replace(
            "class Combined(Mixin, DraftedModel):\n    pass\n\n\n", ""
        ) + "\n\nclass Combined(Mixin, DraftedModel):\n    pass\n"
        models_path.write_text(text)
    _write(
        tmp_path,
        "pkg/server.py",
        "from pkg.models import *\n\n\n"
        f"def serve({consumer_sig}) -> int:\n    {consumer_body}",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=models_path.as_posix(),
            qualname="DraftedModel.running_router",
        )
    )
    (report,) = operation.preflight_reports(snapshot)
    assert report.status is CodemodPreflightStatus.FAILED, report
    assert expected_fragment in report.message


def test_isinstance_narrowed_supertype_receiver_inlines_and_disjoint_sibling_stays(
    tmp_path: Path,
) -> None:
    """When flow-sensitive `isinstance` narrows a superclass-typed local alias
    to `DraftedModel`, the call is rewritten cleanly; a branch narrowed to a
    disjoint sibling subclass stays untouched."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/models.py",
        "class Reader:\n    def running_router(self) -> int:\n        return 7\n\n\n"
        "class Model:\n    pass\n\n\n"
        "class DraftedModel(Model):\n"
        "    def __init__(self, reader: Reader) -> None:\n        self.reader = reader\n\n"
        "    def running_router(self) -> int:\n        return self.reader.running_router()\n\n\n"
        "class OtherModel(Model):\n"
        "    def running_router(self) -> int:\n        return 9\n",
    )
    _write(
        tmp_path,
        "pkg/server.py",
        "from pkg.models import DraftedModel, Model, OtherModel\n\n\n"
        "class Server:\n"
        "    def __init__(self, model: Model) -> None:\n        self.model = model\n\n"
        "    def serve(self) -> int:\n"
        "        active = self.model\n"
        "        if isinstance(active, DraftedModel):\n"
        "            return active.running_router()\n"
        "        if isinstance(active, OtherModel):\n"
        "            return active.running_router()\n"
        "        return 0\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = InlineForwarderOperation(
        target=SourceRewriteTarget(
            file_path=(tmp_path / "pkg/models.py").as_posix(),
            qualname="DraftedModel.running_router",
        )
    )
    simulation = RefactorRecipe("inline").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    simulation.apply()
    models_text = (tmp_path / "pkg/models.py").read_text()
    server_text = (tmp_path / "pkg/server.py").read_text()
    assert "class DraftedModel(Model):\n    def __init__" in models_text
    assert "return active.reader.running_router()" in server_text
    assert "if isinstance(active, OtherModel):\n            return active.running_router()" in server_text
    ran = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pkg.models import DraftedModel, OtherModel, Reader; "
            "from pkg.server import Server; "
            "assert Server(DraftedModel(Reader())).serve() == 7; "
            "assert Server(OtherModel()).serve() == 9",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr


