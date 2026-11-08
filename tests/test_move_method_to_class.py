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


def _simulate(root: Path, qualname: str = "Thread.environment", receiver: str = "thread"):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = MoveMethodToClassOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/threads.py").as_posix(), qualname=qualname),
        destination=SourceRewriteTarget(file_path=(root / "pkg/requests.py").as_posix(), qualname="Request"),
        receiver_parameter=receiver,
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
    assert "    @classmethod\n    def environment(cls, thread, root: str)" in requests
    assert "cls.for_thread(thread)" in requests and "import Request" not in requests
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


@pytest.mark.native_proof
def test_a_method_taking_the_destination_becomes_its_instance_method(tmp_path: Path) -> None:
    threads = THREADS.replace(
        "    def label(self) -> str:\n",
        "    def stamp(self, request: Request, suffix: str) -> str:\n"
        "        return f'{request.thread}:{self.name}{suffix}'\n\n"
        "    def label(self) -> str:\n",
    ).replace("from dataclasses import dataclass, replace\n", (
        "from __future__ import annotations\n\nfrom dataclasses import dataclass, replace\n"
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg.requests import Request\n"
    ))
    config = CONFIG.replace("from pkg.threads import NativeThread, Thread\n", (
        "from pkg.requests import Request\nfrom pkg.threads import NativeThread, Thread\n"
    )) + (
        "\n\ndef stamped(thread: Thread, request: Request) -> str:\n"
        "    return thread.stamp(request, suffix='!')\n"
    )
    _write(tmp_path, {"threads": threads, "requests": REQUESTS, "config": config})
    result = _simulate(tmp_path, "Thread.stamp", receiver="owner")

    assert result.is_clean is True
    requests = _source(tmp_path, result, "requests")
    assert "    def stamp(self, owner, suffix: str) -> str:\n" in requests
    assert "f'{self.thread}:{owner.name}{suffix}'" in requests
    assert "return request.stamp(thread, suffix='!')" in _source(tmp_path, result, "config")
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from pkg.config import stamped; from pkg.requests import Request; "
            "from pkg.threads import NativeThread; print(stamped(NativeThread('w'), Request('r')))"
        )],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "r:w!"


def test_a_method_that_neither_takes_nor_builds_the_destination_is_not_its(tmp_path: Path) -> None:
    _write(tmp_path, {"threads": THREADS, "requests": REQUESTS, "config": CONFIG})

    with pytest.raises(CodemodOperationPreflightError, match="neither takes a Request nor builds one"):
        _simulate(tmp_path, "Thread.label")


IDENTITY = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Identity:\n"
    "    session: str\n"
    "    file: str\n\n"
    "    @staticmethod\n"
    "    def from_proof(proof: 'Proof') -> Identity:\n"
    "        return Identity(proof.session, proof.file)\n\n"
    "    def require(self, proof) -> None:\n"
    "        if self != self.from_proof(proof):\n"
    "            raise ValueError('changed')\n\n"
    "    @classmethod\n"
    "    def from_evidence(cls, evidence: 'Evidence') -> Identity:\n"
    "        \"\"\"Read the identity its evidence names.\"\"\"\n"
    "        return cls(evidence.header(), evidence.path)\n\n"
    "    @classmethod\n"
    "    def proof_for(cls, session: str):\n"
    "        from pkg.proof import Proof\n\n"
    "        return Proof(session, 'f')\n"
)
PROOF = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Proof:\n"
    "    session: str\n"
    "    file: str\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Evidence:\n"
    "    path: str\n\n"
    "    def header(self) -> str:\n"
    "        return 's-' + self.path\n"
)
USE = (
    "from pkg.identity import Identity\n"
    "from pkg.proof import Evidence, Proof\n\n\n"
    "def run() -> str:\n"
    "    proof = Proof('s', 'f')\n"
    "    one = Identity.from_proof(proof)\n"
    "    one.require(proof)\n"
    "    two = Identity.from_evidence(Evidence('p'))\n"
    "    three = Identity.proof_for('x')\n"
    "    return f'{one.session}/{two.session}/{three.session}'\n"
)


def _class_bound(root: Path, qualname: str, destination: str, **fields: str):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = MoveMethodToClassOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/identity.py").as_posix(), qualname=qualname),
        destination=SourceRewriteTarget(file_path=(root / "pkg/proof.py").as_posix(), qualname=destination),
        **fields,
    )
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="move").with_operation(replayed).simulate(snapshot)


def _runs(root: Path, result) -> str:
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)
    return subprocess.run(
        [sys.executable, "-c", "from pkg.use import run; print(run())"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "qualname,destination,new_name,member,call",
    (
        ("Identity.from_proof", "Proof", "identity",
         "    def identity(self) -> Identity:\n        return Identity(self.session, self.file)\n",
         "one = proof.identity()"),
        ("Identity.from_evidence", "Evidence", "identity",
         "    def identity(self) -> Identity:\n        \"\"\"Read the identity its evidence names.\"\"\"\n"
         "        return Identity(self.header(), self.path)\n",
         "two = Evidence('p').identity()"),
        ("Identity.proof_for", "Proof", None,
         "    @classmethod\n    def proof_for(cls, session: str):\n        return cls(session, 'f')\n",
         "three = Proof.proof_for('x')"),
    ),
)
def test_a_static_or_class_method_moves_onto_the_class_it_takes_or_builds(
    tmp_path: Path, qualname: str, destination: str, new_name: str | None, member: str, call: str,
) -> None:
    _write(tmp_path, {"identity": IDENTITY, "proof": PROOF, "use": USE})
    fields = {} if new_name is None else {"destination_name": new_name}
    result = _class_bound(tmp_path, qualname, destination, **fields)

    assert result.is_clean is True
    assert member in _source(tmp_path, result, "proof")
    assert call in _source(tmp_path, result, "use")
    assert qualname.split(".")[1] not in _source(tmp_path, result, "identity").split("def require")[0]
    assert _runs(tmp_path, result) == "s/s-p/x"


def test_a_static_method_call_inside_its_class_moves_with_it(tmp_path: Path) -> None:
    _write(tmp_path, {"identity": IDENTITY, "proof": PROOF, "use": USE})
    result = _class_bound(tmp_path, "Identity.from_proof", "Proof", destination_name="identity")

    assert "if self != proof.identity():" in _source(tmp_path, result, "identity")


@pytest.mark.parametrize(
    "identity,qualname,fields,refusal",
    (
        (IDENTITY + "\n\nclass Child(Identity):\n    pass\n\n\nCHILD = Child.from_evidence\n",
         "Identity.from_evidence", {"destination_name": "identity"}, "other than by calling it"),
        (IDENTITY + "\n\nclass Child(Identity):\n    @classmethod\n    def read(cls, evidence):\n"
         "        return cls.from_evidence(evidence)\n", "Identity.from_evidence",
         {"destination_name": "identity"}, "through cls or self, which may bind cls to a descendant"),
        (IDENTITY + "\n\nclass Child(Identity):\n    pass\n\n\ndef read(evidence):\n"
         "    return Child.from_evidence(evidence)\n", "Identity.from_evidence",
         {"destination_name": "identity"}, "on a class named outright, which may bind cls"),
        (IDENTITY, "Identity.from_proof", {"receiver_parameter": "identity"},
         "is a staticmethod; receiver_parameter is not taken"),
    ),
)
def test_moving_a_class_bound_method_refuses_what_would_change(
    tmp_path: Path, identity: str, qualname: str, fields: dict[str, str], refusal: str,
) -> None:
    destination = "Evidence" if "evidence" in qualname else "Proof"
    _write(tmp_path, {"identity": identity, "proof": PROOF, "use": USE})

    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _class_bound(tmp_path, qualname, destination, **fields)


@pytest.mark.native_proof
def test_descendants_only_matter_where_a_call_could_bind_cls_to_one(tmp_path: Path) -> None:
    identity = IDENTITY + "\n\nclass Child(Identity):\n    pass\n"
    _write(tmp_path, {"identity": identity, "proof": PROOF, "use": USE})

    reads_cls = _class_bound(tmp_path, "Identity.from_evidence", "Evidence", destination_name="identity")
    unread_cls = _class_bound(tmp_path, "Identity.proof_for", "Proof")

    assert "two = Evidence('p').identity()" in _source(tmp_path, reads_cls, "use")
    assert "three = Proof.proof_for('x')" in _source(tmp_path, unread_cls, "use")
