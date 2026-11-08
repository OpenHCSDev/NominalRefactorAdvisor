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
        ("class Other:\n    def require_may_change(self, todo):\n        pass\n", "declares another"),
        ("def bound(actor):\n    return actor.require_may_change\n", "other than by calling it"),
        ("def used(actor, todo):\n    return actor.require_may_change(todo)\n", "returns None"),
        ("def effect(make, todo):\n    make().require_may_change(todo.pop())\n", "reorders the evaluation"),
    ),
)
def test_inlining_by_name_is_refused_when_the_name_is_not_closed(
    tmp_path: Path, consumer: str, message: str
) -> None:
    _identity(tmp_path)
    _write(tmp_path, "pkg/orders.py", "class Todo:\n    pass\n")
    _write(tmp_path, "pkg/consumer.py", consumer)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    report = _inline(tmp_path).preflight_reports(snapshot)
    assert report and report[0].status is CodemodPreflightStatus.FAILED, report
    assert message in report[0].message


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
    _write(tmp_path, "pkg/consumer.py", "def run(actor, effect):\n    actor.require_may_change(effect())\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    report = _inline(tmp_path).preflight_reports(snapshot)
    assert report[0].status is CodemodPreflightStatus.FAILED
    assert "exactly once" in report[0].message
