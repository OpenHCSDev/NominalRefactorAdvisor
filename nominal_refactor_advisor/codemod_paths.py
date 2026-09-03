"""Source-path matching and resolution for codemod plans."""

from __future__ import annotations

from dataclasses import dataclass
from functools import (
    cached_property,
    lru_cache,
)
from pathlib import Path

from .source_identity import (
    resolved_source_path_text,
    source_path_text,
)
from .source_index import SourceIndex


class ExactSourcePathResolution:
    """Resolve an indexed source path exactly as provided by the DSL."""

    @staticmethod
    def matching_paths(
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        return tuple(
            candidate for candidate in projection.paths if candidate == requested_path
        )


class NormalizedSourcePathResolution(ExactSourcePathResolution):
    """Preserve exact resolution and add slash-normalized matching."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        exact_matches = super().matching_paths(requested_path, projection)
        if exact_matches:
            return exact_matches
        requested_posix = source_path_text(requested_path)
        return tuple(
            candidate
            for candidate, candidate_posix in projection.normalized_rows
            if candidate_posix == requested_posix
        )


class ResolvedSourcePathResolution(NormalizedSourcePathResolution):
    """Preserve textual matching and add current-directory resolution."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        textual_matches = super().matching_paths(requested_path, projection)
        if textual_matches:
            return textual_matches
        requested_resolved = resolved_source_path_text(requested_path)
        return tuple(
            candidate
            for candidate, candidate_resolved in projection.resolved_rows
            if candidate_resolved == requested_resolved
        )


class RelativeSuffixSourcePathResolution(ResolvedSourcePathResolution):
    """Preserve stronger matches and add repo-relative suffix resolution."""

    @classmethod
    def matching_paths(
        cls,
        requested_path: str,
        projection: "SourcePathCandidateSet",
    ) -> tuple[str, ...]:
        resolved_matches = super().matching_paths(requested_path, projection)
        if resolved_matches:
            return resolved_matches
        requested = Path(requested_path)
        suffix = f"/{requested.as_posix()}"
        return tuple(
            candidate
            for candidate, candidate_posix in projection.normalized_rows
            if not requested.is_absolute() and candidate_posix.endswith(suffix)
        )


@dataclass(frozen=True)
class SourcePathCandidateSet:
    """Reusable source-index candidate path set with derived projections."""

    paths: tuple[str, ...]

    @classmethod
    def from_paths(
        cls,
        candidate_paths: tuple[str, ...],
    ) -> "SourcePathCandidateSet":
        del cls
        return _source_path_candidate_set(candidate_paths)

    @cached_property
    def normalized_rows(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (candidate, source_path_text(candidate)) for candidate in self.paths
        )

    @cached_property
    def resolved_rows(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (candidate, resolved_source_path_text(candidate))
            for candidate in self.paths
        )


@lru_cache(maxsize=128)
def _source_path_candidate_set(
    candidate_paths: tuple[str, ...],
) -> SourcePathCandidateSet:
    return SourcePathCandidateSet(tuple(sorted(set(candidate_paths))))


@dataclass(frozen=True)
class SourcePathCandidateAuthority:
    """Base authority for resolving DSL paths against indexed source files."""

    requested_path: str
    candidate_set: SourcePathCandidateSet

    @classmethod
    def from_source_index(
        cls,
        requested_path: str,
        source_index: SourceIndex,
    ) -> "SourcePathResolutionAuthority":
        return cls(
            requested_path=requested_path,
            candidate_set=SourcePathCandidateSet.from_paths(
                source_index.target_file_paths
            ),
        )


@dataclass(frozen=True)
class SourcePathResolutionAuthority(SourcePathCandidateAuthority):
    """Resolve DSL file_path values against indexed source files."""

    def optional_path(self) -> str | None:
        matches = self.matching_paths()
        if matches[1:]:
            return None
        return (matches + (None,))[0]

    def required_path(self) -> str:
        matches = self.matching_paths()
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Source path {self.requested_path!r} did not resolve to any "
                "indexed source file"
            )
        raise ValueError(
            f"Source path {self.requested_path!r} resolved to multiple indexed "
            f"source files: {matches!r}"
        )

    def matching_paths(self) -> tuple[str, ...]:
        return RelativeSuffixSourcePathResolution.matching_paths(
            self.requested_path,
            self.candidate_set,
        )


@dataclass(frozen=True)
class SourceCreationPathAuthority(SourcePathCandidateAuthority):
    """Resolve a new DSL file path against existing indexed source roots."""

    def required_path(self) -> str:
        requested = Path(self.requested_path)
        if requested.is_absolute():
            return requested.as_posix()
        parent_matches = self.parent_matches(requested)
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise ValueError(
                f"New source path {self.requested_path!r} resolved to multiple "
                f"candidate locations: {parent_matches!r}"
            )
        return requested.as_posix()

    def parent_matches(self, requested: Path) -> tuple[str, ...]:
        requested_parent = requested.parent.as_posix()
        if requested_parent in ("", "."):
            return ()
        suffix = f"/{requested_parent}"
        return tuple(
            sorted(
                {
                    (Path(candidate).parent / requested.name).as_posix()
                    for candidate in self.candidate_set.paths
                    if Path(candidate).parent.as_posix() == requested_parent
                    or Path(candidate).parent.as_posix().endswith(suffix)
                }
            )
        )
