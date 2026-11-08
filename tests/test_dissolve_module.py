"""A module that declares nothing is dissolved: its consumers follow each name to
the module that declares it, strings naming it follow it, and the file is gone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPreflightStatus,
    CodemodSourceSnapshot,
    DissolveModuleOperation,
    RefactorRecipe,
    RelocateSymbolsToModuleOperation,
    RelocateSymbolsToNewModuleOperation,
    SourceRewriteTarget,
)


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root))


def _run(root: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _dissolve(root: Path, source: str, successor: str | None = None) -> DissolveModuleOperation:
    return DissolveModuleOperation(
        target=SourceRewriteTarget(file_path=(root / source).as_posix()),
        successor_path=None if successor is None else (root / successor).as_posix(),
    )


def test_a_reexport_hub_dissolves_into_its_owners(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/owner.py", "class Moved:\n    value = 3\n")
    _write(tmp_path, "pkg/other.py", "class Kept:\n    value = 4\n")
    _write(
        tmp_path,
        "pkg/hub.py",
        '"""Every tool, in one place."""\n\nfrom .owner import Moved\nfrom .other import Kept\n',
    )
    _write(
        tmp_path,
        "pkg/consumer.py",
        "from pkg import hub\n"
        "from .hub import Kept\n\n"
        "TARGET = 'pkg.hub.Moved'\n"
        "value = hub.Moved.value + Kept.value\n",
    )
    simulation = (
        RefactorRecipe("dissolve-hub")
        .with_operation(_dissolve(tmp_path, "pkg/hub.py"))
        .simulate(_snapshot(tmp_path))
    )

    assert simulation.is_clean
    hub_path = (tmp_path / "pkg/hub.py").as_posix()
    assert simulation.simulation.deleted_file_paths == (hub_path,)
    assert hub_path in simulation.simulation.changed_file_paths
    assert "+++ /dev/null" in simulation.unified_diff(_snapshot(tmp_path).sources_by_file_path)
    consumer = simulation.simulation.rewritten_sources[(tmp_path / "pkg/consumer.py").as_posix()]
    assert "hub" not in consumer.replace("'pkg.owner.Moved'", "")
    assert "'pkg.owner.Moved'" in consumer

    simulation.apply()

    assert not (tmp_path / "pkg/hub.py").exists()
    imported = _run(
        tmp_path,
        "from pkg.consumer import TARGET, value; "
        "assert value == 7 and TARGET == 'pkg.owner.Moved'",
    )
    assert imported.returncode == 0, imported.stderr


def test_an_emptied_module_dissolves_after_relocation_with_its_successor(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/sub/__init__.py", "")
    _write(tmp_path, "pkg/source.py", '"""Orders."""\n\nclass Order:\n    pass\n')
    _write(
        tmp_path,
        "pkg/consumer.py",
        "from .source import Order\n\n"
        "MODULES = ('pkg.source', 'pkg/source.py')\n",
    )
    destination = (tmp_path / "pkg/sub/orders.py").as_posix()
    document = CodemodPlanDocument(
        recipes=(
            RefactorRecipe("relocate-orders").with_operation(
                RelocateSymbolsToNewModuleOperation(
                    target=SourceRewriteTarget(file_path=(tmp_path / "pkg/source.py").as_posix()),
                    symbol_qualnames=("Order",),
                    destination_path=destination,
                )
            ),
        )
    )
    first = document.simulate(_snapshot(tmp_path))
    assert first.is_clean
    first.apply()

    second = (
        RefactorRecipe("dissolve-source")
        .with_operation(_dissolve(tmp_path, "pkg/source.py", "pkg/sub/orders.py"))
        .simulate(_snapshot(tmp_path))
    )
    assert second.is_clean
    second.apply()

    assert not (tmp_path / "pkg/source.py").exists()
    imported = _run(
        tmp_path,
        "from pkg.consumer import Order, MODULES; "
        "assert Order.__module__ == 'pkg.sub.orders'; "
        "assert MODULES == ('pkg.sub.orders', 'pkg/sub/orders.py'), MODULES",
    )
    assert imported.returncode == 0, imported.stderr


@pytest.mark.parametrize(
    ("hub", "consumer", "message"),
    (
        (
            "from .owner import Moved\n\n\nclass Declared:\n    pass\n",
            "from .hub import Moved\n",
            "still declares",
        ),
        (
            "from .owner import Moved\n",
            "def acquire():\n    from . import hub  # noqa: F401\n",
            "for its import side effect",
        ),
        (
            "from .owner import Moved\n",
            "NAME = 'pkg.hub'\n",
            "declare its successor_path",
        ),
        (
            "from dataclasses import dataclass\nfrom .owner import Moved\n",
            "from .hub import dataclass\n",
            "declares none of them",
        ),
    ),
)
def test_dissolution_refuses_a_consumer_with_no_owner_to_follow(
    tmp_path: Path,
    hub: str,
    consumer: str,
    message: str,
) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/owner.py", "class Moved:\n    pass\n")
    _write(tmp_path, "pkg/hub.py", hub)
    _write(tmp_path, "pkg/consumer.py", consumer)

    reports = _dissolve(tmp_path, "pkg/hub.py").preflight_reports(_snapshot(tmp_path))

    assert reports[0].status is CodemodPreflightStatus.FAILED
    assert message in reports[0].message


def test_relocation_redirects_a_function_local_import(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/source.py", "class Kept:\n    pass\n\n\nclass Moved:\n    value = 3\n")
    _write(
        tmp_path,
        "pkg/consumer.py",
        "def read():\n"
        "    from pkg.source import Kept, Moved\n"
        "    return Moved.value, Kept\n",
    )
    simulation = (
        RefactorRecipe("relocate-moved")
        .with_operation(
            RelocateSymbolsToNewModuleOperation(
                target=SourceRewriteTarget(file_path=(tmp_path / "pkg/source.py").as_posix()),
                symbol_qualnames=("Moved",),
                destination_path=(tmp_path / "pkg/moved.py").as_posix(),
            )
        )
        .simulate(_snapshot(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()

    consumer = (tmp_path / "pkg/consumer.py").read_text()
    assert "    from .moved import Moved\n    from pkg.source import Kept\n" in consumer
    imported = _run(tmp_path, "from pkg.consumer import read; assert read()[0] == 3")
    assert imported.returncode == 0, imported.stderr


def _promote(root: Path, source: str, module: str, *names: str):
    from nominal_refactor_advisor.codemod import PromoteGuardedImportOperation

    return PromoteGuardedImportOperation(
        target=SourceRewriteTarget(file_path=(root / source).as_posix()),
        module_name=module,
        import_names=names,
    )


def test_a_guard_hiding_no_cycle_becomes_a_runtime_import(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/owner.py", "class Moved:\n    pass\n")
    _write(
        tmp_path,
        "pkg/reader.py",
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.owner import Moved\n\n\n"
        "def read(moved: Moved) -> Moved:\n"
        "    return moved\n",
    )
    operation = _promote(tmp_path, "pkg/reader.py", "pkg.owner", "Moved")
    assert operation.preflight_reports(_snapshot(tmp_path))[0].status is (
        CodemodPreflightStatus.PASSED
    )
    simulation = RefactorRecipe("promote").with_operation(operation).simulate(_snapshot(tmp_path))
    assert simulation.is_clean
    simulation.apply()

    reader = (tmp_path / "pkg/reader.py").read_text()
    assert "if TYPE_CHECKING" not in reader
    assert "from pkg.owner import Moved" in reader
    imported = _run(tmp_path, "import typing, pkg.reader; assert typing.get_type_hints(pkg.reader.read)")
    assert imported.returncode == 0, imported.stderr


def test_a_guard_hiding_a_cycle_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/owner.py", "from pkg.reader import read\n\n\nclass Moved:\n    pass\n")
    _write(
        tmp_path,
        "pkg/reader.py",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.owner import Moved\n\n\n"
        "def read(moved: 'Moved') -> None:\n"
        "    pass\n",
    )
    report = _promote(tmp_path, "pkg/reader.py", "pkg.owner", "Moved").preflight_reports(
        _snapshot(tmp_path)
    )[0]
    assert report.status is CodemodPreflightStatus.FAILED
    assert "hides a cycle" in report.message


def test_a_move_drops_the_destinations_guarded_import_of_the_moved_declaration(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/frame.py",
        "from pkg.rules import Rule\n\n\nclass Frame:\n    rule = Rule\n",
    )
    _write(
        tmp_path,
        "pkg/rules.py",
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.frame import Frame\n\n\n"
        "class Rule:\n"
        "    def found(self, frame: Frame) -> None:\n"
        "        pass\n",
    )
    simulation = (
        RefactorRecipe("frame-beside-rules")
        .with_operation(
            RelocateSymbolsToModuleOperation(
                target=SourceRewriteTarget(file_path=(tmp_path / "pkg/frame.py").as_posix()),
                symbol_qualnames=("Frame",),
                destination_path=(tmp_path / "pkg/rules.py").as_posix(),
            )
        )
        .simulate(_snapshot(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()

    rules = (tmp_path / "pkg/rules.py").read_text()
    assert "TYPE_CHECKING" not in rules.replace("from typing import TYPE_CHECKING", "")
    assert "class Frame" in rules
    imported = _run(
        tmp_path,
        "import typing, pkg.rules; "
        "assert typing.get_type_hints(pkg.rules.Rule.found)['frame'] is pkg.rules.Frame",
    )
    assert imported.returncode == 0, imported.stderr
