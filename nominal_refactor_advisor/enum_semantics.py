"""Nominal authority for standard-library enum declaration families."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PythonEnumBaseAuthority:
    """Recognize enum bases from their resolved or source-level names."""

    base_names: frozenset[str] = frozenset(
        ("Enum", "Flag", "IntEnum", "IntFlag", "StrEnum")
    )

    def matches(self, base_name: str | None) -> bool:
        return (
            base_name is not None
            and base_name.rsplit(".", maxsplit=1)[-1] in self.base_names
        )

    def matches_any(self, base_names: Iterable[str | None]) -> bool:
        return any(self.matches(base_name) for base_name in base_names)


PYTHON_ENUM_BASE_AUTHORITY = PythonEnumBaseAuthority()
