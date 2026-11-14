"""The advisor's command line, run inside the test's own process.

A test that drives the CLI pays for the work it asks for, not for a fresh
interpreter importing the whole package before it starts (1.8 s each, about 300
times a suite). Tests that need a real process, for a hard exit or their own
environment, still start one.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor import cli

COMMAND = "nominal-refactor-advisor"


def run_advisor(
    *args: str | Path,
    cwd: str | Path | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI on `args` as `python -m nominal_refactor_advisor` would, and
    return its exit code and output as a finished process does."""

    argv = [COMMAND, *(str(arg) for arg in args)]
    stdout, stderr = io.StringIO(), io.StringIO()
    previous_argv, previous_stdin, previous_cwd = sys.argv, sys.stdin, os.getcwd()
    sys.argv, sys.stdin = argv, io.StringIO(input or "")
    if cwd is not None:
        os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                returncode = cli.main()
            except SystemExit as exited:
                match exited.code:
                    case int() as code:
                        returncode = code
                    case None:
                        returncode = 0
                    case message:
                        print(message, file=sys.stderr)
                        returncode = 1
    finally:
        sys.argv, sys.stdin = previous_argv, previous_stdin
        os.chdir(previous_cwd)
    return subprocess.CompletedProcess(argv, returncode, stdout.getvalue(), stderr.getvalue())
