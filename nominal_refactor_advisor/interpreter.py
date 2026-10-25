"""The running interpreter's identities, declared once for every cache key.

Each is a tuple, so entries pickled before these types existed still compare
equal to the running value.
"""

from __future__ import annotations

import sys
from typing import NamedTuple, Self


class PythonVersion(NamedTuple):
    """Major and minor version an AST cache entry was parsed under."""

    major: int
    minor: int

    @classmethod
    def running(cls) -> Self:
        return cls(sys.version_info.major, sys.version_info.minor)


class CompilerIdentity(NamedTuple):
    """Implementation and version that select a native creation backend."""

    implementation: str
    version: PythonVersion

    @classmethod
    def running(cls) -> Self:
        return cls(sys.implementation.name, PythonVersion.running())


class InterpreterBuild(NamedTuple):
    """Exact interpreter build a native compilation ran under."""

    implementation: str
    version: str

    @classmethod
    def running(cls) -> Self:
        return cls(sys.implementation.name, sys.version)
