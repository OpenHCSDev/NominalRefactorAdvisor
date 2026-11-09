"""A package move scans only the files it can read, found from raw text."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    DomainSourcePathPolicy,
    PythonSourcePathPolicy,
)
from nominal_refactor_advisor.refactors.package_move import DestinationTable


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def _admitted(root: Path, table: str) -> frozenset[str]:
    policy = DestinationTable.from_texts((table,)).scan_domain(
        PythonSourcePathPolicy(), (root,)
    )
    assert isinstance(policy, DomainSourcePathPolicy)
    return frozenset(path.relative_to(root.resolve()).as_posix() for path in policy.admitted)


def test_the_domain_holds_consumers_reexport_chains_and_packages(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/messages.py": "from pkg.records import Record\n\nclass Message:\n    pass\n",
            "pkg/records.py": "from pkg.base import Record\n",
            "pkg/base.py": "class Record:\n    pass\n",
            "pkg/consumer.py": "from pkg.messages import Message\n",
            "pkg/unrelated.py": "VALUE = 1\n",
        },
    )

    admitted = _admitted(tmp_path, "| `messaging` | `messages.py` | |\n")

    assert {
        "pkg/messages.py",
        "pkg/consumer.py",
        "pkg/records.py",
        "pkg/base.py",
        "pkg/__init__.py",
        "pkg/messaging/__init__.py",
    } <= admitted
    assert "pkg/unrelated.py" not in admitted


def test_a_row_naming_a_class_seeds_the_module_declaring_it(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/planning/__init__.py": "",
            "pkg/goals.py": "class GoalWait:\n    pass\n",
            "pkg/reader.py": "from pkg.goals import GoalWait\n",
            "pkg/unrelated.py": "VALUE = 1\n",
        },
    )

    admitted = _admitted(tmp_path, "| `planning` | GoalWait and its waits | |\n")

    assert {"pkg/goals.py", "pkg/reader.py"} <= admitted
    assert "pkg/unrelated.py" not in admitted


def test_a_table_with_no_rows_scans_every_file(tmp_path: Path) -> None:
    policy = PythonSourcePathPolicy()

    assert DestinationTable.from_texts(("no table here",)).scan_domain(policy, (tmp_path,)) is policy
