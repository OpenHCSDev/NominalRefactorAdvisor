"""A scanned project's declaration that nothing outside it calls into it.

A public function is open to callers the scan cannot see, so no proof over
the scan can say every call to it is known. An application, unlike a library,
can declare that it has no outside callers, in its ``pyproject.toml``::

    [tool.nominal-refactor-advisor]
    closed-world = true

The premise closes only that: outside callers. A function that escapes as a
value, or that a computed fetch can return, stays open. A proof that relies on
the premise records the declaring file and its content digest, so a changed or
removed declaration is a different premise; without one, public functions stay
open.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .declared_root import DeclaredRoot


@dataclass(frozen=True)
class ClosedWorldPremise:
    """The declaring ``pyproject.toml`` and the sha256 of its bytes when it was read."""

    pyproject: str
    digest: str

    @classmethod
    def declared_for(cls, source: Path) -> ClosedWorldPremise | None:
        pyproject = DeclaredRoot.nearest_pyproject(source)
        return None if pyproject is None else cls.declared_in(pyproject)

    @classmethod
    def declared_in(cls, pyproject: Path) -> ClosedWorldPremise | None:
        status = pyproject.stat()
        return cls.read(pyproject, status.st_mtime_ns, status.st_size)

    @classmethod
    @cache
    def read(
        cls, pyproject: Path, modified_ns: int, size: int
    ) -> ClosedWorldPremise | None:
        """The premise in one version of the file; a rewrite is read again."""

        del modified_ns, size
        content = pyproject.read_bytes()
        match tomllib.loads(content.decode()):
            case {"tool": {"nominal-refactor-advisor": {"closed-world": True}}}:
                return cls(str(pyproject), hashlib.sha256(content).hexdigest())
        return None
