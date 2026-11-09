"""A destination table's moves run as one relocate stage and one dissolve stage:
every moved block is rewritten through the whole batch's redirect, and every
consumer once."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import ArchitectureGuardSuite, CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.refactor_concepts import PackageMoveConcept


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _moved(root: Path, table: str) -> tuple[int, Path]:
    report = CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(package_destination_tables=(table,)),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert report.stop_reason.completed, report.to_markdown()
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    )
    assert replay.is_clean
    replay.apply()
    return len(report.replay_sequence.documents), root


def _runs(root: Path, expression: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", f"from pkg.app import VALUE; print({expression})"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_a_moved_block_importing_another_moved_module_lands_at_its_new_home(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/channels.py": "class Channel:\n    name = 'all'\n",
            "pkg/channel_targets.py": (
                "from __future__ import annotations\n\n"
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n    from pkg.channels import Channel\n\n\n"
                "class Target:\n    def of(self, channel: Channel) -> str:\n"
                "        return channel.name\n"
            ),
            "pkg/app.py": (
                "from pkg.channel_targets import Target\nfrom pkg.channels import Channel\n\n"
                "VALUE = Target().of(Channel())\n"
            ),
        },
    )

    stages, root = _moved(
        tmp_path, "| `messaging` | `channels.py`, `channel_targets.py` | |\n"
    )

    assert stages == 2
    assert sorted(path.name for path in (root / "pkg").glob("*.py")) == ["__init__.py", "app.py"]
    moved = (root / "pkg/messaging/channel_targets.py").read_text()
    assert "from pkg.messaging.channels import Channel" in moved
    assert "pkg.channels" not in moved
    assert _runs(root, "VALUE") == "all"


def test_a_reexport_chain_across_two_moved_modules_resolves_from_the_start(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/records.py": "class Record:\n    kind = 'record'\n",
            "pkg/messages.py": (
                "from pkg.records import Record\n\n__all__ = ('Message', 'Record')\n\n\n"
                "class Message(Record):\n    pass\n"
            ),
            "pkg/app.py": "from pkg.messages import Message, Record\n\nVALUE = (Message.kind, Record.kind)\n",
        },
    )

    stages, root = _moved(tmp_path, "| `messaging` | `messages.py`, `records.py` | |\n")

    assert stages == 2
    assert sorted(path.name for path in (root / "pkg").glob("*.py")) == ["__init__.py", "app.py"]
    assert "pkg.records" not in (root / "pkg/app.py").read_text()
    assert _runs(root, "VALUE") == "('record', 'record')"


def test_a_module_importing_a_peer_bare_moves_one_module_per_stage(tmp_path: Path) -> None:
    """`import pkg.records` is read as `pkg.records.Record`: the batch cannot rename
    it in the moved text, so the table stages instead and still lands."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/records.py": "class Record:\n    kind = 'record'\n",
            "pkg/messages.py": (
                "import pkg.records\n\n\nclass Message(pkg.records.Record):\n    pass\n"
            ),
            "pkg/app.py": "from pkg.messages import Message\n\nVALUE = Message.kind\n",
        },
    )

    stages, root = _moved(tmp_path, "| `messaging` | `messages.py`, `records.py` | |\n")
    assert stages > 2
    assert _runs(root, "VALUE") == "record"


def test_an_aliased_peer_import_inside_a_function_follows_the_batch(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/records.py": "class Record:\n    kind = 'record'\n",
            "pkg/messages.py": (
                "def kind() -> str:\n"
                "    import pkg.records as records\n\n"
                "    return records.Record.kind\n"
            ),
            "pkg/app.py": "from pkg.messages import kind\n\nVALUE = kind()\n",
        },
    )

    stages, root = _moved(tmp_path, "| `messaging` | `messages.py`, `records.py` | |\n")
    assert stages == 2
    assert "import pkg.messaging.records as records" in (
        root / "pkg/messaging/messages.py"
    ).read_text()
    assert _runs(root, "VALUE") == "record"


def test_a_peer_imported_from_its_parent_package_is_imported_from_its_new_one(
    tmp_path: Path,
) -> None:
    """`from pkg import records`, with records a peer the batch moves, imports it
    from its new package: at module level (re-derived as a dependency) and inside a
    function (rewritten in the moved text)."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/records.py": "class Record:\n    kind = 'record'\n",
            "pkg/messages.py": (
                "from pkg import records\n\n\n"
                "class Message(records.Record):\n    pass\n\n\n"
                "def late_kind() -> str:\n"
                "    from pkg import records as late\n\n"
                "    return late.Record.kind\n"
            ),
            "pkg/app.py": (
                "from pkg.messages import Message, late_kind\n\n"
                "VALUE = (Message.kind, late_kind())\n"
            ),
        },
    )

    stages, root = _moved(tmp_path, "| `messaging` | `messages.py`, `records.py` | |\n")
    assert stages == 2
    moved = (root / "pkg/messaging/messages.py").read_text()
    assert "from pkg.messaging import records\n" in moved
    assert "from pkg.messaging import records as late" in moved
    assert _runs(root, "VALUE") == "('record', 'record')"
