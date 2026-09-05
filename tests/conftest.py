"""Repository-wide pytest environment ownership."""

from __future__ import annotations

import os
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.detectors import _base as collector_runtime
from nominal_refactor_advisor.source_geometry import read_source_text


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
