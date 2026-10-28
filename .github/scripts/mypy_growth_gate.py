"""Growth-only mypy gate: no file may gain type errors against the base commit.

The baseline is the base commit's own mypy run, computed here, never a
committed list. Both trees are checked with the same mypy and flags, so a
count can only move because the source moved.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "nominal_refactor_advisor"
FLAGS = (
    "--strict-equality",
    "--ignore-missing-imports",
    "--python-version",
    "3.11",
    "--no-error-summary",
    "--no-pretty",
)
ERROR = re.compile(r"^(?P<path>[^:]+):\d+(?::\d+)?: error: .*\[(?P<code>[a-z-]+)\]$")


@dataclass(frozen=True)
class MypyRun:
    """Error counts of one tree, by file and by error code."""

    by_file: Counter[str]
    by_code: Counter[str]

    @classmethod
    def of(cls, tree: Path, cache: Path) -> MypyRun:
        completed = subprocess.run(
            [sys.executable, "-m", "mypy", PACKAGE, *FLAGS, "--cache-dir", str(cache)],
            cwd=tree,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise SystemExit(f"mypy failed on {tree}:\n{completed.stderr}")
        by_file: Counter[str] = Counter()
        by_code: Counter[str] = Counter()
        for line in completed.stdout.splitlines():
            if (match := ERROR.match(line)) is not None:
                by_file[match["path"]] += 1
                by_code[match["code"]] += 1
        return cls(by_file, by_code)

    @property
    def total(self) -> int:
        return sum(self.by_file.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref", required=True, help="commit the head may not regress"
    )
    arguments = parser.parse_args()
    head_tree = Path.cwd()
    with tempfile.TemporaryDirectory() as scratch:
        base_tree = Path(scratch) / "base"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(base_tree), arguments.base_ref],
            check=True,
            capture_output=True,
        )
        try:
            base = MypyRun.of(base_tree, Path(scratch) / "cache-base")
            head = MypyRun.of(head_tree, Path(scratch) / "cache-head")
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(base_tree)], check=False
            )
    print(f"mypy errors: base {base.total} -> head {head.total}")
    for code in sorted(
        set(base.by_code) | set(head.by_code), key=lambda c: -base.by_code[c]
    ):
        if base.by_code[code] or head.by_code[code]:
            print(f"  {code:20} {base.by_code[code]:5} -> {head.by_code[code]}")
    grown = sorted(
        (path, base.by_file[path], count)
        for path, count in head.by_file.items()
        if count > base.by_file[path]
    )
    for path, before, after in grown:
        print(f"::error file={path}::mypy errors grew {before} -> {after}")
    return 1 if grown else 0


if __name__ == "__main__":
    raise SystemExit(main())
