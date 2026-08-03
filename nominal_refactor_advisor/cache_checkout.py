"""Safe checkout-relative identities for relocatable derived caches."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


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
    lexical_parents = tuple(str(_lexical_absolute_path(path).parent) for path in paths)
    common_parent = Path(os.path.commonpath(lexical_parents))
    return (common_parent,)


def checkout_relative_path(
    path: Path | str,
    roots: tuple[Path | str, ...],
) -> str:
    """Encode one path by its unique ordered checkout root and safe relative path."""

    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        _validate_relative_text(candidate_path.as_posix())
        if len(roots) != 1:
            raise CacheCheckoutPathError(
                f"relative path {candidate_path} is ambiguous across {len(roots)} roots"
            )
        lexical_root = _lexical_absolute_path(roots[0])
        lexical_path = (
            lexical_root
            if lexical_root.is_file() and candidate_path == Path(lexical_root.name)
            else _lexical_absolute_path(lexical_root / candidate_path)
        )
    else:
        lexical_path = _lexical_absolute_path(candidate_path)
    matches: list[tuple[int, Path]] = []
    for root_index, root_value in enumerate(roots):
        lexical_root = _lexical_absolute_path(root_value)
        if lexical_root.is_file():
            if lexical_path == lexical_root:
                matches.append((root_index, Path(".")))
            continue
        try:
            matches.append((root_index, lexical_path.relative_to(lexical_root)))
        except ValueError:
            continue
    if len(matches) != 1:
        reason = (
            "outside every admitted root" if not matches else "matches multiple roots"
        )
        raise CacheCheckoutPathError(f"{lexical_path} {reason}")
    root_index, relative_path = matches[0]
    relative_text = relative_path.as_posix()
    _validate_relative_text(relative_text)
    return f"{root_index}:{relative_text}"


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
    lexical_root = _lexical_absolute_path(roots[root_index])
    if lexical_root.is_file():
        if relative_text != ".":
            raise CacheCheckoutPathError(
                f"file root {lexical_root} cannot admit {relative_text!r}"
            )
        return str(lexical_root)
    lexical_path = _lexical_absolute_path(lexical_root / relative_text)
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


def _lexical_absolute_path(path: Path | str) -> Path:
    """Canonicalize spelling without dereferencing an admitted source symlink."""

    return Path(os.path.abspath(os.fspath(path)))
