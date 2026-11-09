"""In a git worktree the scan's source is git's: every tracked or new unignored
Python file, a hidden directory's included."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.test_package_move_batch import _moved, _write


def test_a_move_rewrites_a_tracked_script_in_a_hidden_directory(tmp_path: Path) -> None:
    """`.github/scripts/use.py` is tracked and imports `pkg.store`. The directory
    walk pruned it by its leading dot, so the move left it importing a module the
    move deleted. Read from git, it is a consumer like any other and follows."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/store.py": "class Profiles:\n    pass\n",
            "pkg/app.py": "from pkg.store import Profiles\n\nVALUE = Profiles.__name__\n",
            ".github/scripts/use.py": (
                "from pkg.store import Profiles\n\nprint(Profiles.__name__)\n"
            ),
        },
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    _stages, root = _moved(tmp_path, "| `messaging` | `store.py` | |\n")
    assert not (root / "pkg/store.py").exists()
    assert "from pkg.messaging.store import Profiles" in (
        root / ".github/scripts/use.py"
    ).read_text()
    result = subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path('.github/scripts/use.py')"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Profiles\n"
