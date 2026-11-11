"""Overrides answering issubclass(arg, C) become one declared requirement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    DeclaredCapabilityRequirementOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)

BRIDGES = (
    "class CliBridge:\n    pass\n\n\n"
    "class LineMonitorBridge(CliBridge):\n    pass\n\n\n"
    "class ClaudeBridge(LineMonitorBridge):\n    pass\n\n\n"
    "class PlainBridge(CliBridge):\n    pass\n"
)
OPERATIONS = (
    "from pkg.bridges import CliBridge, ClaudeBridge, LineMonitorBridge\n\n\n"
    "class Operation:\n"
    "    @classmethod\n"
    "    def serves(cls, bridge: type[CliBridge]) -> bool:\n"
    "        \"\"\"Whether this command runs on that bridge; every bridge sends.\"\"\"\n"
    "        return True\n\n"
    "    @classmethod\n"
    "    def offered(cls, bridge: type[CliBridge]) -> list[str]:\n"
    "        return [member.__name__ for member in cls.__subclasses__() if member.serves(bridge)]\n\n\n"
    "class Send(Operation):\n    pass\n\n\n"
    "class Watch(Operation):\n"
    "    @classmethod\n"
    "    def serves(cls, bridge: type[CliBridge]) -> bool:\n"
    "        return issubclass(bridge, LineMonitorBridge)\n\n\n"
    "class Connect(Operation):\n"
    "    help = 'connect'\n\n"
    "    @classmethod\n"
    "    def serves(cls, bridge: type) -> bool:\n"
    "        \"\"\"Only Claude connects.\"\"\"\n"
    "        return issubclass(bridge, ClaudeBridge)\n"
)
PROBE = (
    "from pkg.bridges import ClaudeBridge, PlainBridge\nfrom pkg.operations import Operation\n"
    "print(Operation.offered(ClaudeBridge), Operation.offered(PlainBridge), Operation.serves(int))\n"
)


def _tree(root: Path, operations: str = OPERATIONS) -> DeclaredCapabilityRequirementOperation:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/bridges.py").write_text(BRIDGES)
    (root / "pkg/operations.py").write_text(operations)
    return DeclaredCapabilityRequirementOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/operations.py").as_posix(), qualname="Operation.serves"
        ),
        attribute_name="requires",
    )


def _run(root: Path) -> str:
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def test_each_override_declares_what_it_requires(tmp_path: Path) -> None:
    operation = _tree(tmp_path)
    before = _run(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    simulation = RefactorRecipe("requirement").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    for path, source in simulation.simulation.rewritten_sources.items():
        Path(path).write_text(source)
    source = (tmp_path / "pkg/operations.py").read_text()
    assert "    requires: ClassVar[type] = object\n" in source
    assert "        return issubclass(bridge, cls.requires)\n" in source
    assert "class Watch(Operation):\n    requires = LineMonitorBridge\n" in source
    assert "    requires = ClaudeBridge\n    \"\"\"Only Claude connects.\"\"\"\n" in source
    assert source.count("def serves") == 1
    assert _run(tmp_path) == before == "['Send', 'Watch', 'Connect'] ['Send'] True\n"


def test_a_negated_requirement_is_refused_by_name(tmp_path: Path) -> None:
    operation = _tree(
        tmp_path,
        OPERATIONS + (
            "\n\nclass Inbox(Operation):\n"
            "    @classmethod\n"
            "    def serves(cls, bridge: type[CliBridge]) -> bool:\n"
            "        return not issubclass(bridge, ClaudeBridge)\n"
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="Inbox.serves answers"):
        operation.source_edits(snapshot)
