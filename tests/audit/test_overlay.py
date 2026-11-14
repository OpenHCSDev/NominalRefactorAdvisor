"""The overlay runs every finding kind of its family over a package at a revision."""
from __future__ import annotations

import subprocess
from pathlib import Path

from nominal_refactor_advisor.audit.findings import (
    AtLine,
    Finding,
    ListedFinding,
    Overlay,
    Package,
    PerLineFinding,
)
from nominal_refactor_advisor.audit.repository import Repository

IDENTITY = ("-c", "user.name=Audit test", "-c", "user.email=audit@example.invalid",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null")

ROWS = '''import subprocess


def insert(connection, row):
    # the legacy path
    connection.execute("INSERT INTO rows VALUES (?, ?)", row)
    connection.execute("INSERT INTO rows (a, b) VALUES (?, ?)", row)
    if set(row) == {"a", "b"}:
        return getattr(row, "a", None)
    return subprocess.run(["true"])
'''


def _overlay(root: Path) -> Overlay:
    package = root / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "rows.py").write_text(ROWS)
    for command in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "one")):
        subprocess.run(("git", *IDENTITY, *command), cwd=root, check=True, capture_output=True)
    return Overlay.run(Package.load(Repository(root), "HEAD", "pkg"))


def test_the_overlay_runs_every_finding_kind(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path)

    assert list(overlay.findings) == list(Finding.members())
    counts = {kind.family_name: len(found) for kind, found in overlay.findings.items()}
    for name in (
        "legacy_marker",
        "positional_insert",
        "column_insert",
        "exact_key_set",
        "named_attribute",
        "child_process_site",
    ):
        assert counts[name] == 1, name
    (marker,) = overlay.of(Finding.decode("legacy_marker"))
    assert isinstance(marker, AtLine)
    assert (marker.location, marker.file) == ("pkg/rows.py:5", "pkg/rows.py")
    assert overlay.render(10, overlay.baseline()).startswith("SUMMARY")


def test_a_class_that_groups_findings_is_not_a_finding_kind() -> None:
    kinds = set(Finding.members())

    assert kinds.isdisjoint({AtLine, ListedFinding, PerLineFinding})
    assert set(ListedFinding.members()) < kinds
