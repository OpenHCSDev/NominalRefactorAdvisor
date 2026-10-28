"""Repository-wide pytest environment ownership."""

from __future__ import annotations

import os
import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleSet
from nominal_refactor_advisor.detectors import _base as collector_runtime
from nominal_refactor_advisor.interpreter import CompilerIdentity
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.source_geometry import read_source_text


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        help="Run only shard I of N, chosen by a stable hash of each test id.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "native_proof: the test's subject is a proof that only a native creation "
        "backend with a contract for the running interpreter can give",
    )


def _native_proof_marks() -> tuple[pytest.MarkDecorator, ...]:
    """The marks a `native_proof` test gets: a skip when the running
    interpreter's backend is not a contracted member of NativeCreationBackend's
    registry (SpanOnlyCreationBackend), none when it is."""

    backend = NativeCreationBackend.current()
    if type(backend) in NativeCreationBackend.__registry__.values():
        return ()
    return (
        pytest.mark.skip(
            reason=f"{type(backend).__name__}: no native creation contract for "
            f"{CompilerIdentity.running()}; the proof under test cannot be given"
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    marks = _native_proof_marks()
    for item in items:
        if item.get_closest_marker("native_proof") is not None:
            for mark in marks:
                item.add_marker(mark)
    shard = config.getoption("--shard")
    if shard is None:
        return
    index, count = (int(part) for part in shard.split("/"))
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        digest = hashlib.sha256(item.nodeid.encode()).digest()
        (kept if int.from_bytes(digest[:8]) % count == index else deselected).append(
            item
        )
    config.hook.pytest_deselected(items=deselected)
    items[:] = kept


@pytest.fixture(scope="session")
def native_collector_module() -> ParsedModule:
    """Current native collector declarations, shared without copying their schema."""

    path = Path(collector_runtime.__file__)
    source = read_source_text(path)
    return ParsedModule(
        path=path,
        module_name=collector_runtime.__name__,
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


@pytest.fixture(scope="session")
def no_modules() -> ParsedModuleSet:
    """The module set of a run that parsed nothing."""

    return ParsedModuleSet(())


@pytest.fixture(scope="session", autouse=True)
def _isolated_default_advisor_cache() -> Iterator[None]:
    """Keep test-created persistent caches inside one disposable session root."""

    environment_name = "NRA_CACHE_HOME"
    previous_cache_home = os.environ.get(environment_name)
    with TemporaryDirectory(prefix="nra-pytest-cache-") as cache_home:
        os.environ[environment_name] = cache_home
        try:
            yield
        finally:
            if previous_cache_home is None:
                os.environ.pop(environment_name, None)
            else:
                os.environ[environment_name] = previous_cache_home


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """`-n auto` takes the cores nobody else is using.

    The suite shares its machine with other agents' runs. One-minute load average counts
    the runnable processes already competing, so the free cores are the CPUs minus that
    load, never fewer than one worker; `--maxprocesses` still caps the result."""
    return max(1, (os.cpu_count() or 1) - round(os.getloadavg()[0]))
