"""Safe checkout-relative identities for relocatable derived caches."""

from __future__ import annotations

import io
import os
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePath, PurePosixPath
from typing import BinaryIO

from nominal_refactor_advisor.scan_cache import ScanCache


@dataclass(frozen=True)
class CheckoutRoot:
    """Current filesystem facts needed by lexical checkout admission.

    Capture is deliberately uncached: changing a root between file and directory
    must invalidate an otherwise identical path query within the same scan.
    """

    path: Path
    is_file: bool

    @classmethod
    def capture(cls, root: Path | str) -> CheckoutRoot:
        path = lexical_absolute_path(root)
        return cls(path, path.is_file())

    def relative_path(self, candidate: Path) -> Path | None:
        if self.is_file:
            return Path(".") if candidate == self.path else None
        try:
            return candidate.relative_to(self.path)
        except ValueError:
            return None


@ScanCache.cached
def _admitted_checkout_path(candidate: Path, roots: tuple[CheckoutRoot, ...]) -> str:
    """Pure admission under the complete, currently observed ordered root set."""
    matches = tuple(
        (index, relative)
        for index, root in enumerate(roots)
        if (relative := root.relative_path(candidate)) is not None
    )
    if len(matches) != 1:
        reason = (
            "outside every admitted root" if not matches else "matches multiple roots"
        )
        raise CacheCheckoutPathError(f"{candidate} {reason}")
    root_index, relative_path = matches[0]
    relative_text = relative_path.as_posix()
    _validate_relative_text(relative_text)
    return f"{root_index}:{relative_text}"


@ScanCache.cached
def _absolute_path_value(absolute_text: str) -> Path:
    """Reuse immutable spelling only after the caller resolves its current cwd."""
    return Path(absolute_text)


class CacheCheckoutPathError(ValueError):
    """A cached path cannot be proved to belong to exactly one admitted root."""


def presentation_root_texts(roots: tuple[Path | str, ...]) -> tuple[str, ...]:
    """Return canonical absolute presentation roots without giving them identity."""

    return tuple(str(Path(root).resolve()) for root in roots)


def semantic_root_labels(roots: tuple[Path | str, ...]) -> tuple[str, ...]:
    """Represent only ordered root slots in semantic cache identity."""

    return tuple(f"root:{index}" for index, _root in enumerate(roots))


def inferred_checkout_roots(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Infer one presentation root for APIs that are not given scan roots."""

    if not paths:
        return ()
    lexical_parents = tuple(str(lexical_absolute_path(path).parent) for path in paths)
    common_parent = Path(os.path.commonpath(lexical_parents))
    return (common_parent,)


def checkout_relative_path(
    path: Path | str,
    roots: tuple[Path | str, ...],
) -> str:
    """Encode a path under the current complete root set, preserving lexical links."""
    candidate_path = Path(path)
    is_relative = not candidate_path.is_absolute()
    if is_relative:
        _validate_relative_text(candidate_path.as_posix())
        if len(roots) != 1:
            raise CacheCheckoutPathError(
                f"relative path {candidate_path} is ambiguous across {len(roots)} roots"
            )
    observed_roots = tuple(CheckoutRoot.capture(root) for root in roots)
    if is_relative:
        (root,) = observed_roots
        lexical_path = (
            root.path
            if root.is_file
            and (
                candidate_path == Path(root.path.name)
                or lexical_absolute_path(candidate_path) == root.path
            )
            else lexical_absolute_path(root.path / candidate_path)
        )
    else:
        lexical_path = lexical_absolute_path(candidate_path)
    return _admitted_checkout_path(lexical_path, observed_roots)


def absolute_checkout_path(
    logical_path: str,
    roots: tuple[Path | str, ...],
) -> str:
    """Resolve a validated logical cache path under its current presentation root."""

    root_index, relative_text = _parse_logical_path(logical_path)
    if root_index >= len(roots):
        raise CacheCheckoutPathError(
            f"logical root {root_index} is absent from {len(roots)} admitted roots"
        )
    lexical_root = lexical_absolute_path(roots[root_index])
    if lexical_root.is_file():
        if relative_text != ".":
            raise CacheCheckoutPathError(
                f"file root {lexical_root} cannot admit {relative_text!r}"
            )
        return str(lexical_root)
    lexical_path = lexical_absolute_path(lexical_root / relative_text)
    try:
        lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise CacheCheckoutPathError(
            f"logical path {logical_path!r} escapes {lexical_root}"
        ) from error
    return str(lexical_path)


def rebase_checkout_path(
    path: Path | str,
    source_roots: tuple[Path | str, ...],
    target_roots: tuple[Path | str, ...],
) -> str:
    """Rebase an admitted source path onto equivalent ordered target roots."""

    if len(source_roots) != len(target_roots):
        raise CacheCheckoutPathError(
            "cached and requested checkout root counts do not match"
        )
    logical_path = checkout_relative_path(path, source_roots)
    if presentation_root_texts(source_roots) == presentation_root_texts(target_roots):
        return str(path)
    return absolute_checkout_path(logical_path, target_roots)


def _parse_logical_path(logical_path: str) -> tuple[int, str]:
    root_text, separator, relative_text = logical_path.partition(":")
    if not separator or not root_text.isdecimal():
        raise CacheCheckoutPathError(f"invalid logical cache path {logical_path!r}")
    _validate_relative_text(relative_text)
    return int(root_text), relative_text


def _validate_relative_text(relative_text: str) -> None:
    if not relative_text:
        raise CacheCheckoutPathError("empty checkout-relative path")
    relative_path = PurePosixPath(relative_text)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CacheCheckoutPathError(f"unsafe checkout-relative path {relative_text!r}")
    if relative_text != relative_path.as_posix():
        raise CacheCheckoutPathError(
            f"non-canonical checkout-relative path {relative_text!r}"
        )


def lexical_absolute_path(path: Path | str) -> Path:
    """Canonicalize current spelling without dereferencing a source symlink."""
    return _absolute_path_value(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True)
class CheckoutRelocation:
    """One checkout's paths, stored relative to it so every checkout of the tree reads them.

    A checkout is the nearest directory holding `.git` (a linked worktree holds a
    `.git` file). Derived state about a module names the module's own checkout as
    strings and paths beginning with the checkout root; stored relative and rebased
    on load, an entry computed in one worktree is the entry every other worktree at
    the same content would compute. A payload that still spells the root after
    relocation (inside a longer string, say) is not portable and stays bound to
    the checkout that wrote it.
    """

    root: Path

    @classmethod
    def containing(cls, path: Path | str) -> CheckoutRelocation | None:
        return _checkout_containing(str(lexical_absolute_path(path).parent))

    @cached_property
    def root_texts(self) -> tuple[str, ...]:
        """Lexical spelling first, then the resolved one when a link differs."""

        lexical = str(self.root)
        resolved = str(self.root.resolve())
        return (lexical,) if lexical == resolved else (lexical, resolved)

    def path_within(self, path: Path | str) -> str:
        """The path from the checkout's root, as every worktree spells it."""

        return lexical_absolute_path(path).relative_to(self.root).as_posix()

    def relative_text(self, path: Path | str) -> str:
        return f"checkout:{self.path_within(path)}"

    def relative_id(self, value: object) -> tuple[object, ...] | None:
        if type(value) is str:
            text, kind = value, None
        elif isinstance(value, PurePath):
            text, kind = str(value), type(value)
        else:
            return None
        for root in self.root_texts:
            if text.startswith(root) and (
                len(text) == len(root) or text[len(root)] == "/"
            ):
                return ("checkout", kind, text[len(root) :])
        return None

    def restored(self, relative_id: tuple[object, ...]) -> object:
        marker, kind, suffix = relative_id
        if marker != "checkout" or not isinstance(suffix, str):
            raise pickle.UnpicklingError("unknown persistent checkout reference")
        text = self.root_texts[0] + suffix
        return text if kind is None else kind(text)

    def is_mentioned_in(self, data: bytes) -> bool:
        return any(root.encode("utf-8") in data for root in self.root_texts)

    def portable_dumps(self, value: object) -> tuple[bytes, bool]:
        """The relocated pickle, or the plain one when relocation leaves the root."""

        buffer = io.BytesIO()
        _RelocatingPickler(buffer, self).dump(value)
        data = buffer.getvalue()
        if not self.is_mentioned_in(data):
            return data, True
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), False

    def load(self, handle: BinaryIO) -> object:
        return _RelocatingUnpickler(handle, self).load()


class _RelocatingPickler(pickle.Pickler):
    def __init__(self, handle: BinaryIO, relocation: CheckoutRelocation) -> None:
        super().__init__(handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._relocation = relocation

    def persistent_id(self, obj: object) -> tuple[object, ...] | None:
        return self._relocation.relative_id(obj)


class _RelocatingUnpickler(pickle.Unpickler):
    def __init__(self, handle: BinaryIO, relocation: CheckoutRelocation) -> None:
        super().__init__(handle)
        self._relocation = relocation

    def persistent_load(self, pid: object) -> object:
        if not isinstance(pid, tuple) or len(pid) != 3:
            raise pickle.UnpicklingError("unknown persistent reference")
        return self._relocation.restored(pid)


def checkout_identity_text(path: Path | str) -> str:
    """A path as identity: where it sits in its checkout, else as spelled. A
    relative path names no location, so it is its own identity."""

    if not Path(path).is_absolute():
        return str(path)
    checkout = CheckoutRelocation.containing(path)
    return str(path) if checkout is None else checkout.relative_text(path)


def checkout_display_path(path: Path | str) -> str:
    """A path as prose names it: from its checkout's root, else as spelled. A
    summary that spells a checkout's directory is bound to that checkout."""

    if not Path(path).is_absolute():
        return str(path)
    checkout = CheckoutRelocation.containing(path)
    return str(path) if checkout is None else checkout.path_within(path)


@ScanCache.cached
def _checkout_containing(directory_text: str) -> CheckoutRelocation | None:
    directory = Path(directory_text)
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return CheckoutRelocation(candidate)
    return None
