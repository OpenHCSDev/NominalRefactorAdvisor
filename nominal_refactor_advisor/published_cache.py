"""Publishing a cache file so a concurrent reader sees it whole or not at all."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class PublishesCacheFiles:
    """A cache whose entries are published atomically.

    `published(path)` writes through a private sibling file and renames it into
    place: a concurrent reader opens the old entry or the new one, never a
    partial write, because the bytes reach disk before the rename. A failed or
    interrupted write leaves the old entry and no stray temporary behind. A cache
    that needs the rename itself durable overrides `_sync_published_directory`."""

    __slots__ = ()

    @contextmanager
    def published(self, path: Path) -> Iterator[BinaryIO]:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staged = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                yield handle
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise
        self._sync_published_directory(path.parent)

    def _sync_published_directory(self, directory: Path) -> None:
        """Make the rename itself durable; a no-op where that is not required."""
