"""Provider history repairs callers without importing apps or editing providers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.audit.repository import Repository
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    ReplayGitModuleImportsOperation,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=NRA test", "-c", "user.email=nra@example.invalid", *args],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def recorded_move(tmp_path: Path) -> tuple[Path, str, str, Path]:
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "pkg").mkdir()
    (provider / "pkg/__init__.py").write_text('raise RuntimeError("provider was imported during planning")\n')
    (provider / "pkg/old.py").write_text("VALUE = 17\ndef answer():\n    return VALUE\n")
    git(provider, "init", "-q")
    git(provider, "add", ".")
    git(provider, "commit", "-qm", "original provider")
    before = git(provider, "rev-parse", "HEAD")
    (provider / "pkg/nominal").mkdir()
    (provider / "pkg/nominal/__init__.py").write_text("")
    git(provider, "mv", "pkg/old.py", "pkg/nominal/new.py")
    git(provider, "add", ".")
    git(provider, "commit", "-qm", "move provider")
    after = git(provider, "rev-parse", "HEAD")
    caller = tmp_path / "caller"
    caller.mkdir()
    return provider, before, after, caller


@pytest.mark.parametrize("source", [
    "from pkg.old import answer as call\nprint(call())\n",
    "def run():\n    from pkg.old import answer as call\n    return call()\nprint(run())\n",
    "import pkg.old as provider\nprint(provider.answer())\n",
    "import sys\nfrom pkg.old import answer as call\nprint(call())\n",
])
def test_replay_uses_recorded_moves_and_only_writes_callers(recorded_move, source: str) -> None:
    provider, before, after, caller = recorded_move
    path = caller / "consumer.py"
    path.write_text(source)
    unrelated = caller.parent / "unrelated.py"
    unrelated.write_text("from pkg.old import answer\n")
    operation = ReplayGitModuleImportsOperation(
        provider_repository=str(provider), source_revision=before,
        target_revision="HEAD", caller_root=str(caller),
    )
    assert operation.target_revision == after
    tree = Repository(provider).tree(after)
    result = RefactorRecipe("replay-importing-callers").with_operation(operation).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(caller.parent))
    )
    assert result.is_clean, result
    assert set(result.simulation.rewritten_sources) == {str(path)}
    result.apply()
    assert "pkg.old" not in path.read_text()
    assert path.read_text() == source.replace("pkg.old", "pkg.nominal.new")
    assert unrelated.read_text() == "from pkg.old import answer\n"
    assert Repository(provider).tree(after) == tree
    assert git(provider, "status", "--porcelain") == ""
    # The same real caller runs against the target provider after the native edit.
    (provider / "pkg/__init__.py").write_text("")
    environment = {**os.environ, "PYTHONPATH": str(provider)}
    run = subprocess.run([sys.executable, str(path)], env=environment,
                         capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    assert run.stdout == "17\n"


def test_replay_refuses_missing_target_bindings(recorded_move) -> None:
    provider, before, _after, caller = recorded_move
    # Keep Git's detected move, but remove one of its required answers.
    (provider / "pkg/nominal/new.py").write_text("def answer():\n    return VALUE\n")
    git(provider, "add", ".")
    git(provider, "commit", "-qm", "remove answer")
    (caller / "consumer.py").write_text("from pkg.old import VALUE\n")
    operation = ReplayGitModuleImportsOperation(
        provider_repository=str(provider), source_revision=before,
        target_revision="HEAD", caller_root=str(caller),
    )
    with pytest.raises(CodemodOperationPreflightError, match="loses provider bindings"):
        RefactorRecipe("refuse-missing-binding").with_operation(operation).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(caller))
        )


def test_replay_requires_an_isolated_caller_tree(recorded_move) -> None:
    provider, before, after, _caller = recorded_move
    with pytest.raises(ValueError, match="must be separate"):
        ReplayGitModuleImportsOperation(
            provider_repository=str(provider), source_revision=before,
            target_revision=after, caller_root=str(provider),
        )
