"""Repository-wide pytest environment ownership."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest


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
