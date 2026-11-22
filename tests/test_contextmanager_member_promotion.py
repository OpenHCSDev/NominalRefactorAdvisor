"""Promote the original lazy read without executing its generator at definition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PromoteClassMembersToAncestorOperation,
    SourceRewriteTarget,
)


def test_promotes_contextmanager_without_changing_binding_or_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "publication.py"
    path.write_text(
        "from contextlib import contextmanager\n"
        "events = []\n"
        "class Publication:\n"
        "    pass\n"
        "class Read(Publication):\n"
        "    @classmethod\n"
        "    @contextmanager\n"
        "    def publication_source(cls, root):\n"
        "        events.append(('enter', root))\n"
        "        try:\n"
        "            yield cls.capture(root)\n"
        "        finally:\n"
        "            events.append(('exit', root))\n"
        "    @classmethod\n"
        "    def capture(cls, root):\n"
        "        return cls.__name__, root\n"
        "class Child(Read):\n"
        "    pass\n"
        "assert not events\n"
        "with Child.publication_source('normal') as value:\n"
        "    assert value == ('Child', 'normal')\n"
        "try:\n"
        "    with Read.publication_source('failure'):\n"
        "        raise RuntimeError('original')\n"
        "except RuntimeError as error:\n"
        "    assert str(error) == 'original'\n"
        "else:\n"
        "    raise AssertionError('exception suppressed')\n"
        "print(events)\n",
        encoding="utf-8",
    )
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    operation = PromoteClassMembersToAncestorOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Read"),
        destination=SourceRewriteTarget(file_path=str(path), qualname="Publication"),
        member_names=("publication_source", "capture"),
    )
    result = CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert result.is_clean
    result.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
