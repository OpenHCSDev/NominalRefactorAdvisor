from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.refactors.member_move import MoveMethodToClassOperation

THREADS = (
    "from dataclasses import dataclass, replace\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Thread:\n"
    "    name: str\n\n"
    "    def environment(self, root: str) -> dict[str, str]:\n"
    "        from pkg.requests import Request\n\n"
    "        request = Request.for_thread(self)\n"
    "        return {**request.encoded(root), 'name': replace(self, name=self.name).name}\n\n"
    "    def label(self) -> str:\n"
    "        return self.name\n\n\n"
    "class NativeThread(Thread):\n"
    "    pass\n"
)
REQUESTS = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n\n"
    "from pkg.threads import Thread\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Request:\n"
    "    thread: str\n\n"
    "    @classmethod\n"
    "    def for_thread(cls, thread: Thread) -> Request:\n"
    "        return cls(thread.name)\n\n"
    "    def encoded(self, root: str) -> dict[str, str]:\n"
    "        return {'request': f'{root}/{self.thread}'}\n"
)
CONFIG = (
    "from pkg.threads import NativeThread, Thread\n\n\n"
    "class Config:\n"
    "    def environment(self, thread: Thread) -> dict[str, str]:\n"
    "        return {'config': 'on', **thread.environment('/root')}\n\n\n"
    "class Owner:\n"
    "    def __init__(self, thread: NativeThread, config: Config) -> None:\n"
    "        self.thread = thread\n"
    "        self.config = config\n\n"
    "    def launch(self) -> dict[str, str]:\n"
    "        return self.config.environment(self.thread)\n"
)


def _write(root: Path, sources: dict[str, str]) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    for name, source in sources.items():
        (root / f"pkg/{name}.py").write_text(source)


def _simulate(root: Path, qualname: str = "Thread.environment"):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = MoveMethodToClassOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/threads.py").as_posix(), qualname=qualname),
        destination=SourceRewriteTarget(file_path=(root / "pkg/requests.py").as_posix(), qualname="Request"),
        receiver_parameter="thread",
    )
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="move").with_operation(replayed).simulate(snapshot)


def _source(root: Path, result, module: str) -> str:
    return result.simulation.rewritten_sources[(root / f"pkg/{module}.py").as_posix()]


@pytest.mark.native_proof
def test_a_method_moves_onto_the_class_it_builds_and_only_its_receivers_follow(tmp_path: Path) -> None:
    _write(tmp_path, {"threads": THREADS, "requests": REQUESTS, "config": CONFIG})
    result = _simulate(tmp_path)

    assert result.is_clean is True
    threads = _source(tmp_path, result, "threads")
    assert "def environment" not in threads and "pkg.requests" not in threads
    assert "def label" in threads
    requests = _source(tmp_path, result, "requests")
    assert "    @staticmethod\n    def environment(thread, root: str)" in requests
    assert "Request.for_thread(thread)" in requests and "import Request" not in requests
    config = _source(tmp_path, result, "config")
    assert "Request.environment(thread, '/root')" in config
    assert "return self.config.environment(self.thread)" in config
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from pkg.config import Config, Owner; from pkg.threads import NativeThread; "
            "print(Owner(NativeThread('w'), Config()).launch())"
        )],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "{'config': 'on', 'request': '/root/w', 'name': 'w'}"


@pytest.mark.parametrize(
    "sources,refusal",
    (
        ({"threads": THREADS + "\n\nclass Loud(Thread):\n    def environment(self, root: str) -> dict[str, str]:\n        return {}\n",
          "requests": REQUESTS, "config": CONFIG}, "override 'environment'"),
        ({"threads": THREADS, "requests": REQUESTS,
          "config": CONFIG + "\n\nNAME = 'environment'\n"}, "names 'environment' in a string"),
        ({"threads": THREADS, "requests": REQUESTS,
          "config": CONFIG + "\n\ndef pick(thread: Thread):\n    return thread.environment\n"},
         "other than by calling it"),
    ),
)
def test_moving_a_method_refuses_what_its_callers_would_lose(
    tmp_path: Path, sources: dict[str, str], refusal: str,
) -> None:
    _write(tmp_path, sources)

    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _simulate(tmp_path)
