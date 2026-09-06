"""Canonical source-file identities shared by analysis and codemod projections."""

from __future__ import annotations

import hashlib

from abc import ABC
from collections.abc import Mapping
from functools import cached_property
from os import PathLike, fspath
from pathlib import Path


def python_source_cache_signature(source: str) -> str:
    """Return the exact UTF-8 source identity shared by analysis and compilation."""
    return hashlib.blake2s(source.encode("utf-8"), digest_size=16).hexdigest()


class SourceFileIdentity(ABC):
    """Nominal owner of canonical source-path projections."""

    path: Path

    @cached_property
    def file_path(self) -> str:
        """Canonical slash-normalized source identity for cross-platform joins."""

        return source_path_text(self.path)

    @cached_property
    def resolved_file_path(self) -> str:
        """Canonical absolute source identity for filesystem comparisons."""

        return resolved_source_path_text(self.path)


def source_path_text(path: str | PathLike[str]) -> str:
    """Return the platform-independent textual identity of one source path."""

    return fspath(path).replace("\\", "/")


def resolved_source_path_text(path: str | PathLike[str]) -> str:
    """Return the absolute platform-independent identity of one source path."""

    return Path(path).expanduser().resolve().as_posix()


def canonical_source_mapping(
    source_by_path: Mapping[str, str],
) -> dict[str, str]:
    """Index source text by canonical path, rejecting ambiguous identities."""

    canonical_sources: dict[str, str] = {}
    original_paths: dict[str, str] = {}
    for file_path, source in source_by_path.items():
        canonical_path = source_path_text(file_path)
        previous_path = original_paths.get(canonical_path)
        if previous_path is not None:
            raise ValueError(
                "Multiple source paths resolve to the same canonical identity: "
                f"{previous_path!r} and {file_path!r}"
            )
        original_paths[canonical_path] = file_path
        canonical_sources[canonical_path] = source
    return canonical_sources
