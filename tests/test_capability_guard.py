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


PHASES = (
    "class MaintenancePhase:\n    pass\n\n\n"
    "class DrainingPhase(MaintenancePhase):\n    pass\n\n\n"
    "class HeldPhase(MaintenancePhase):\n    pass\n\n\n"
    "class MaintenanceState:\n"
    "    phase: type[MaintenancePhase]\n\n"
    "    def __init__(self, phase: type[MaintenancePhase]) -> None:\n        self.phase = phase\n\n"
    "    def require_draining(self) -> None:\n"
    "        if self.phase is not DrainingPhase:\n"
    "            raise ValueError('Only an acquired draining phase can be cleared')\n"
)


def _phases(root: Path, source: str = PHASES) -> CapabilityGuardToOverrideOperation:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/phases.py").write_text(source)
    path = (root / "pkg/phases.py").as_posix()
    return CapabilityGuardToOverrideOperation(
        target=SourceRewriteTarget(file_path=path, qualname="MaintenanceState.require_draining"),
        base=SourceRewriteTarget(file_path=path, qualname="MaintenancePhase"),
        method_name="require_draining",
    )


def test_a_guard_on_a_field_s_class_is_asked_of_that_class(tmp_path: Path) -> None:
    """agent-comms MaintenanceState.require_draining: `self.phase is not DrainingPhase`."""
    probe = (
        "from pkg.phases import DrainingPhase, HeldPhase, MaintenanceState\n"
        "MaintenanceState(DrainingPhase).require_draining()\n"
        "try:\n    MaintenanceState(HeldPhase).require_draining()\nexcept ValueError as error:\n    print(error)\n"
    )
    operation = _phases(tmp_path)
    before = _run(tmp_path, probe)
    _apply(tmp_path, operation)
    source = (tmp_path / "pkg/phases.py").read_text()
    assert "        self.phase.require_draining()\n" in source
    assert " is not " not in source
    assert _run(tmp_path, probe) == before


def test_an_exact_class_test_with_descendants_is_refused(tmp_path: Path) -> None:
    operation = _phases(tmp_path, PHASES + "\n\nclass LateDraining(DrainingPhase):\n    pass\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="descend from it"):
        operation.source_edits(snapshot)


def _messages(root: Path, imports: str, annotation: str) -> CapabilityGuardToOverrideOperation:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/message.py").write_text(
        "class Message:\n    pass\n\n\nclass Close(Message):\n    pass\n\n\nclass Open(Message):\n    pass\n"
    )
    (root / "pkg/messages.py").write_text("from pkg.message import Close, Message, Open\n")
    (root / "pkg/driver.py").write_text(
        f"{imports}from pkg.message import Close\n\n\n"
        f"def close(message: {annotation}) -> str:\n"
        "    if not isinstance(message, Close):\n"
        "        raise ValueError('only a close')\n"
        "    return type(message).__name__\n"
    )
    return CapabilityGuardToOverrideOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/driver.py").as_posix(), qualname="close"),
        base=SourceRewriteTarget(file_path=(root / "pkg/message.py").as_posix(), qualname="Message"),
        method_name="require_close",
    )


@pytest.mark.parametrize(
    ("imports", "annotation"),
    (("from pkg import messages\n", "messages.Message"), ("from pkg.messages import Message\n", "Message")),
)
def test_an_annotation_through_a_re_export_names_its_declaring_class(
    tmp_path: Path, imports: str, annotation: str
) -> None:
    """`messages.Message` and `from pkg.messages import Message`, where `pkg.messages`
    imports `Message` from `pkg.message`, name `pkg.message.Message` (Textual's
    `Driver.process_message(self, message: messages.Message)`)."""
    probe = (
        "from pkg.driver import close\nfrom pkg.message import Close, Open\n"
        "print(close(Close()))\n"
        "try:\n    close(Open())\nexcept ValueError as error:\n    print(error)\n"
    )
    operation = _messages(tmp_path, imports, annotation)
    before = _run(tmp_path, probe)
    _apply(tmp_path, operation)
    assert "    message.require_close()\n" in (tmp_path / "pkg/driver.py").read_text()
    assert _run(tmp_path, probe) == before == "Close\nonly a close\n"


@pytest.mark.parametrize(
    ("returned", "outcome"),
    (
        ("Message", None),
        ("Message | None", "whose values are not all instances"),
        ("int", "which is not one class the scan declares"),
    ),
)
def test_an_unannotated_subject_takes_the_type_checker_s_class(
    tmp_path: Path, returned: str, outcome: str | None
) -> None:
    """A local has no annotation; its type at the guard, from the scan's type-checker
    build, is its declared class. A type with None, or no scanned class, refuses: the
    call would run on values the guard refused."""
    operation = _messages(tmp_path, "from pkg.message import Message\n", "Message")
    driver = tmp_path / "pkg/driver.py"
    driver.write_text(
        driver.read_text()
        .replace("def close(message: Message) -> str:\n", f"def pick(value: object) -> {returned}:\n    return value  # type: ignore[return-value]\n\n\ndef close(value: object) -> str:\n    message = pick(value)\n")
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    if outcome is not None:
        with pytest.raises(CodemodOperationPreflightError, match=outcome):
            operation.source_edits(snapshot)
        return
    _apply(tmp_path, operation)
    assert "    message.require_close()\n" in driver.read_text()
