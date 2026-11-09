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


def test_a_module_bound_from_its_package_follows_as_a_module(tmp_path: Path) -> None:
    """`from pkg import commands as cmds` reads `cmds.ClearQueue` beside another
    `ClearQueue` it binds directly: the import line names the module's new home and
    the reads stay, where redirecting them to a name import would collide."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/commands.py": "class ClearQueue:\n    kind = 'command'\n",
            "pkg/inbox.py": "class ClearQueue:\n    kind = 'inbox'\n",
            "pkg/app.py": (
                "from pkg import commands as cmds\n"
                "from pkg.inbox import ClearQueue\n\n"
                "VALUE = (cmds.ClearQueue.kind, ClearQueue.kind)\n"
            ),
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `commands.py` | |\n")
    app = (root / "pkg/app.py").read_text()
    assert "from pkg.messaging import commands as cmds\n" in app
    assert "cmds.ClearQueue.kind" in app
    assert _runs(root, "VALUE") == "('command', 'inbox')"


def test_a_guarded_module_binding_stays_guarded(tmp_path: Path) -> None:
    """`if TYPE_CHECKING: from pkg import events as ev`, read only in annotations,
    keeps its import under the guard: a runtime import would close the cycle the
    guard exists to break (events imports app at load time)."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/events.py": (
                "from pkg.app import Failure\n\n\n"
                "class Response:\n    failure = Failure\n"
            ),
            "pkg/app.py": (
                "from __future__ import annotations\n\n"
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n    from pkg import events as ev\n\n\n"
                "class Failure:\n    pass\n\n\n"
                "def answer(response: ev.Response) -> ev.Response:\n    return response\n\n\n"
                "VALUE = answer.__name__\n"
            ),
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `events.py` | |\n")
    app = (root / "pkg/app.py").read_text()
    assert "if TYPE_CHECKING:\n    from pkg.messaging import events as ev\n" in app
    assert "def answer(response: ev.Response) -> ev.Response:" in app
    assert _runs(root, "VALUE") == "answer"


def test_a_module_bound_only_to_run_it_follows_the_move(tmp_path: Path) -> None:
    """`import pkg.registry as registry` and `from pkg import registry as reg`, read
    nowhere, import the module for what it runs: they name its new home."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/registry.py": "import pkg.seen\n\nENTRY = pkg.seen.NAMES.append('registered')\n",
            "pkg/seen.py": "NAMES = []\n",
            "pkg/app.py": (
                "import pkg.registry as registry\n"
                "from pkg import registry as reg\n"
                "from pkg.seen import NAMES\n\n"
                "VALUE = NAMES[0]\n"
            ),
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `registry.py` | |\n")
    app = (root / "pkg/app.py").read_text()
    assert "import pkg.messaging.registry as registry\n" in app
    assert "from pkg.messaging import registry as reg\n" in app
    assert _runs(root, "VALUE") == "registered"


def test_a_name_the_moved_module_only_rebinds_is_read_from_its_declaring_module(
    tmp_path: Path,
) -> None:
    """`policy` re-binds `Grant` from `sources` and reads it nowhere; a consumer's
    function-local `from pkg.policy import Grant` reads it from `sources` once
    the move lands, so the re-export dies with the shell."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/sources.py": "class Grant:\n    kind = 'grant'\n",
            "pkg/policy.py": "from pkg.sources import Grant\n\n\nclass Policy:\n    pass\n",
            "pkg/app.py": (
                "def kind() -> str:\n"
                "    from pkg.policy import Grant\n\n"
                "    return Grant.kind\n\n\n"
                "VALUE = kind()\n"
            ),
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `policy.py` | |\n")
    app = (root / "pkg/app.py").read_text()
    assert "from pkg.sources import Grant" in app
    assert not (root / "pkg/policy.py").exists()
    assert _runs(root, "VALUE") == "grant"


def test_moved_modules_named_beside_others_in_one_import_each_follow(tmp_path: Path) -> None:
    """`from pkg import events, kept, wires` names two modules the batch moves and
    one that stays: each moved one leaves the statement for its new package, the
    reads stay as written, and `kept` keeps its import."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/events.py": "class Response:\n    kind = 'event'\n",
            "pkg/wires.py": "class Response:\n    kind = 'wire'\n",
            "pkg/kept.py": "class Response:\n    kind = 'kept'\n",
            "pkg/app.py": (
                "from pkg import events, kept, wires\n\n"
                "VALUE = (events.Response.kind, wires.Response.kind, kept.Response.kind)\n"
            ),
        },
    )

    stages, root = _moved(tmp_path, "| `messaging` | `events.py`, `wires.py` | |\n")
    assert stages == 2
    app = (root / "pkg/app.py").read_text()
    assert "from pkg import kept\n" in app
    assert "events.Response.kind, wires.Response.kind" in app
    assert _runs(root, "VALUE") == "('event', 'wire', 'kept')"


def test_a_whole_module_move_carries_the_statements_between_its_declarations(
    tmp_path: Path,
) -> None:
    """An `assert` on the constant above it and a registration call bind nothing,
    so no declaration names them; a whole module's move carries them after what
    they follow, and the shell left behind dissolves."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/seen.py": "NAMES = []\n",
            "pkg/limits.py": (
                "import pkg.seen\n\n"
                "BUDGET = 3\n"
                "assert BUDGET > 1\n\n\n"
                "class Policy:\n    budget = BUDGET\n\n\n"
                "pkg.seen.NAMES.append(Policy.__name__)\n"
            ),
            "pkg/app.py": (
                "from pkg.limits import Policy\n"
                "from pkg.seen import NAMES\n\n"
                "VALUE = (Policy.budget, NAMES[0])\n"
            ),
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `limits.py` | |\n")
    assert not (root / "pkg/limits.py").exists()
    moved = (root / "pkg/messaging/limits.py").read_text()
    assert moved.index("BUDGET = 3") < moved.index("assert BUDGET > 1") < moved.index("class Policy")
    assert moved.index("class Policy") < moved.index("pkg.seen.NAMES.append(Policy.__name__)")
    assert _runs(root, "VALUE") == "(3, 'Policy')"


def test_a_merge_row_carries_the_statements_of_the_module_it_folds(tmp_path: Path) -> None:
    """`limits.py merges into policy.py` moves every declaration into a module that
    already exists: the source never becomes that module, but its shell must still
    empty, so the assert on its constant moves too."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/planning/__init__.py": "",
            "pkg/policy.py": "class Policy:\n    pass\n",
            "pkg/limits.py": "LIMIT = 2\nassert LIMIT > 1\n",
            "pkg/app.py": (
                "from pkg.limits import LIMIT\nfrom pkg.policy import Policy\n\n"
                "VALUE = (LIMIT, Policy.__name__)\n"
            ),
        },
    )

    _stages, root = _moved(
        tmp_path, "| `planning` | `policy.py`; `limits.py` merges into `policy.py` | |\n"
    )
    assert not (root / "pkg/limits.py").exists()
    merged = (root / "pkg/planning/policy.py").read_text()
    assert merged.index("LIMIT = 2") < merged.index("assert LIMIT > 1")
    assert _runs(root, "VALUE") == "(2, 'Policy')"


def test_a_row_whose_own_move_refuses_leaves_the_batch_and_the_rest_move(
    tmp_path: Path,
) -> None:
    """`gamma`'s destination already declares `Gamma`, so its move refuses on its
    own. Each row is checked where every destination the batch creates exists, so
    the collision is gamma's alone: it stays put with its reason and `alpha`
    and `beta` still move in one stage."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/alpha.py": "class Alpha:\n    kind = 'a'\n",
            "pkg/beta.py": "from pkg.alpha import Alpha\n\n\nclass Beta(Alpha):\n    kind = 'b'\n",
            "pkg/gamma.py": "class Gamma:\n    kind = 'flat'\n",
            "pkg/messaging/gamma.py": "class Gamma:\n    kind = 'nested'\n",
            "pkg/app.py": (
                "from pkg.beta import Beta\nfrom pkg.gamma import Gamma\n\n"
                "VALUE = (Beta.kind, Gamma.kind)\n"
            ),
        },
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(
            package_destination_tables=(
                "| `messaging` | `alpha.py`, `beta.py`, `gamma.py` | |\n",
            )
        ),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert "already binds moved declarations ('Gamma',)" in report.to_markdown()
    assert len(report.replay_sequence.documents) == 2
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert replay.is_clean
    replay.apply()
    assert not (tmp_path / "pkg/alpha.py").exists()
    assert not (tmp_path / "pkg/beta.py").exists()
    assert (tmp_path / "pkg/gamma.py").exists()
    assert _runs(tmp_path, "VALUE") == "('b', 'flat')"



def test_a_batch_peer_import_heads_the_new_module_above_the_block_reading_it(
    tmp_path: Path,
) -> None:
    """`Beta(Alpha)` reads its peer's name when the class is built. The import and
    the block both land on line 1 of the new module, and the batch's edit stream
    opens with alpha's block, so the import has to lead by what it is (an import),
    not by which peer's edit came first."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/alpha.py": "class Alpha:\n    kind = 'a'\n",
            "pkg/beta.py": "from pkg.alpha import Alpha\n\n\nclass Beta(Alpha):\n    kind = 'b'\n",
            "pkg/app.py": "from pkg.beta import Beta\n\nVALUE = Beta.kind\n",
        },
    )

    _stages, root = _moved(tmp_path, "| `messaging` | `alpha.py`, `beta.py` | |\n")
    moved = (root / "pkg/messaging/beta.py").read_text()
    assert moved.index("from pkg.messaging.alpha import Alpha") < moved.index("class Beta")
    assert _runs(root, "VALUE") == "b"


def test_a_module_statement_no_declaration_carries_refuses_its_whole_move(
    tmp_path: Path,
) -> None:
    """`gamma`'s loop fills `NAMES` when gamma loads. Moving `NAMES` would leave the
    loop in a shell nothing imports, so `NAMES` would arrive empty: gamma's move
    refuses with the stranded line, it leaves the batch, and the rest still move."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/alpha.py": "class Alpha:\n    kind = 'a'\n",
            "pkg/beta.py": "class Beta:\n    kind = 'b'\n",
            "pkg/gamma.py": "NAMES = []\nfor n in range(2):\n    NAMES.append(n)\n",
            "pkg/app.py": (
                "from pkg.alpha import Alpha\nfrom pkg.beta import Beta\n"
                "from pkg.gamma import NAMES\n\nVALUE = (Alpha.kind, Beta.kind, NAMES)\n"
            ),
        },
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(
            package_destination_tables=(
                "| `messaging` | `alpha.py`, `beta.py`, `gamma.py` | |\n",
            )
        ),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert "would keep 'for n in range(2):' at line 2" in report.to_markdown()
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert replay.is_clean
    replay.apply()
    assert not (tmp_path / "pkg/alpha.py").exists()
    assert not (tmp_path / "pkg/beta.py").exists()
    assert (tmp_path / "pkg/gamma.py").read_text().startswith("NAMES = []\n")
    assert _runs(tmp_path, "VALUE") == "('a', 'b', [0, 1])"


def test_a_shell_whose_own_dissolve_refuses_stays_and_the_rest_still_move(
    tmp_path: Path,
) -> None:
    """`held` and `plain` only re-export `Alpha` and dissolve before anything
    relocates. Neither has a module to dissolve into, so each refuses on its own;
    that used to refuse the whole dissolve stage and left `alpha` waiting behind
    `held` forever. Each shell now stays with its reason and `alpha` moves, its
    re-exports following it."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/alpha.py": "class Alpha:\n    kind = 'a'\n",
            "pkg/held.py": "from pkg.alpha import Alpha\n",
            "pkg/plain.py": "from pkg.alpha import Alpha\n",
            "pkg/app.py": (
                "import pkg.held as held\nfrom pkg.plain import Alpha\n\n"
                "VALUE = (Alpha.kind, held.Alpha.kind)\n"
            ),
        },
    )

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(
            package_destination_tables=(
                "| `messaging` | `alpha.py`, `held.py`, `plain.py` | |\n",
            )
        ),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert report.to_markdown().count("stays: its dissolve refused") == 2
    replay = report.replay_sequence.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert replay.is_clean
    replay.apply()
    assert not (tmp_path / "pkg/alpha.py").exists()
    assert (tmp_path / "pkg/plain.py").read_text().splitlines() == [
        "from pkg.messaging.alpha import Alpha",
        "",
    ]
    assert _runs(tmp_path, "VALUE") == "('a', 'a')"


def test_a_bare_peer_import_holds_back_its_own_module_not_the_batch(
    tmp_path: Path,
) -> None:
    """`messages` reads its peer as `pkg.records.Record`, which the batch cannot
    rename in moved text. That holds back `messages` alone: `alpha`, `beta` and
    `records` still relocate in one stage, and `messages` follows on its own with
    its import rewritten like any consumer's. One such import used to drop a whole
    table to one module per stage."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/alpha.py": "class Alpha:\n    kind = 'a'\n",
            "pkg/beta.py": "class Beta:\n    kind = 'b'\n",
            "pkg/records.py": "class Record:\n    kind = 'record'\n",
            "pkg/messages.py": (
                "import pkg.records\n\n\nclass Message(pkg.records.Record):\n    pass\n"
            ),
            "pkg/app.py": (
                "from pkg.alpha import Alpha\nfrom pkg.beta import Beta\n"
                "from pkg.messages import Message\n\n"
                "VALUE = (Alpha.kind, Beta.kind, Message.kind)\n"
            ),
        },
    )

    stages, root = _moved(
        tmp_path,
        "| `messaging` | `alpha.py`, `beta.py`, `records.py`, `messages.py` | |\n",
    )
    assert stages == 4
    assert not any((root / f"pkg/{name}.py").exists() for name in ("alpha", "beta", "records", "messages"))
    assert _runs(root, "VALUE") == "('a', 'b', 'record')"
