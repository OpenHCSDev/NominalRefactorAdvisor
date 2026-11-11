"""A guard refusing what lacks a capability becomes the capability's override."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CapabilityGuardToOverrideOperation,
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)

KINDS = (
    "class MessageKind:\n"
    "    declared_name = 'kind'\n\n"
    "    @classmethod\n"
    "    def require_sendable(cls) -> None:\n"
    "        if not issubclass(cls, SendableKind):\n"
    "            raise ValueError(f'{cls.declared_name} is not sendable')\n\n\n"
    "class SendableKind(MessageKind):\n"
    "    declared_name = 'sendable'\n\n\n"
    "class InfoKind(SendableKind):\n    declared_name = 'info'\n\n\n"
    "class ClaimKind(MessageKind):\n    declared_name = 'claim'\n"
)

ACTIONS = (
    "class GoalActor:\n    pass\n\n\n"
    "class OwnerControl(GoalActor):\n    pass\n\n\n"
    "class GoalAction(GoalActor):\n    pass\n\n\n"
    "class Pause(GoalAction, OwnerControl):\n    pass\n\n\n"
    "class Report(GoalAction):\n    pass\n\n\n"
    "def update_goal(action: type[GoalAction]) -> str:\n"
    "    if not issubclass(action, OwnerControl):\n"
    "        raise ValueError('only owner controls')\n"
    "    return action.__name__\n"
)


def _run(root: Path, program: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def _apply(root: Path, operation: CapabilityGuardToOverrideOperation) -> None:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    simulation = RefactorRecipe("guard").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    for path, source in simulation.simulation.rewritten_sources.items():
        Path(path).write_text(source)


PROBE_KINDS = (
    "from pkg.kinds import ClaimKind, InfoKind\n"
    "InfoKind.require_sendable()\n"
    "try:\n    ClaimKind.require_sendable()\nexcept ValueError as error:\n    print(error)\n"
)


def test_a_guard_that_is_the_whole_method_becomes_the_capability_s_override(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/kinds.py").write_text(KINDS)
    before = _run(tmp_path, PROBE_KINDS)
    _apply(
        tmp_path,
        CapabilityGuardToOverrideOperation(
            target=SourceRewriteTarget(
                file_path=(tmp_path / "pkg/kinds.py").as_posix(),
                qualname="MessageKind.require_sendable",
            ),
            base=SourceRewriteTarget(file_path=(tmp_path / "pkg/kinds.py").as_posix(), qualname="MessageKind"),
            method_name="require_sendable",
        ),
    )
    source = (tmp_path / "pkg/kinds.py").read_text()
    assert "issubclass" not in source
    assert "class SendableKind(MessageKind):\n    declared_name = 'sendable'\n\n    @classmethod\n" in source
    assert _run(tmp_path, PROBE_KINDS) == before == "claim is not sendable\n"


def _actions(root: Path, base: str) -> CapabilityGuardToOverrideOperation:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/actions.py").write_text(ACTIONS)
    path = (root / "pkg/actions.py").as_posix()
    return CapabilityGuardToOverrideOperation(
        target=SourceRewriteTarget(file_path=path, qualname="update_goal"),
        base=SourceRewriteTarget(file_path=path, qualname=base),
        method_name="require_owner_control",
    )


def test_a_base_whose_answer_beats_the_mixin_capability_is_refused(tmp_path: Path) -> None:
    operation = _actions(tmp_path, "GoalAction")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="would dispatch"):
        operation.source_edits(snapshot)


def test_a_guard_on_a_parameter_asks_the_capability_s_root(tmp_path: Path) -> None:
    probe = (
        "from pkg.actions import Pause, Report, update_goal\n"
        "print(update_goal(Pause))\n"
        "try:\n    update_goal(Report)\nexcept ValueError as error:\n    print(error)\n"
    )
    operation = _actions(tmp_path, "GoalActor")
    before = _run(tmp_path, probe)
    _apply(tmp_path, operation)
    source = (tmp_path / "pkg/actions.py").read_text()
    assert "    action.require_owner_control()\n    return action.__name__\n" in source
    assert _run(tmp_path, probe) == before == "Pause\nonly owner controls\n"


def test_a_family_member_reaching_no_answer_is_refused(tmp_path: Path) -> None:
    """The bare root can be passed too: without the capability's root in its MRO it
    would raise AttributeError where it raised the guard's error."""
    operation = _actions(tmp_path, "GoalActor")
    (tmp_path / "pkg/actions.py").write_text(ACTIONS.replace("class GoalAction(GoalActor):", "class GoalAction:"))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="reaches neither"):
        operation.source_edits(snapshot)
