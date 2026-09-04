"""Nominal authority for standard-library enum declaration families."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, Flag, IntEnum, IntFlag, StrEnum


@dataclass(frozen=True)
class PythonEnumBaseAuthority:
    """Recognize enum bases from their resolved or source-level names."""

    base_names: frozenset[str] = frozenset(
        ("Enum", "Flag", "IntEnum", "IntFlag", "StrEnum")
    )
    inherited_member_names: frozenset[str] = frozenset(
        member_name
        for enum_type in (Enum, Flag, IntEnum, IntFlag, StrEnum)
        for ancestor_type in enum_type.__mro__
        for member_name in vars(ancestor_type)
    )

    def matches(self, base_name: str | None) -> bool:
        return (
            base_name is not None
            and base_name.rsplit(".", maxsplit=1)[-1] in self.base_names
        )

    def matches_any(self, base_names: Iterable[str | None]) -> bool:
        return any(self.matches(base_name) for base_name in base_names)

    def matches_qualified(self, qualified_name: str | None) -> bool:
        """Recognize only declarations resolved to the standard enum module."""

        if qualified_name is None:
            return False
        module_name, separator, base_name = qualified_name.rpartition(".")
        return (
            separator == "." and module_name == "enum" and base_name in self.base_names
        )

    def permits_new_member(self, member_name: str) -> bool:
        """Reject additions that would replace standard enum behavior."""

        return member_name not in self.inherited_member_names

    def declared_member_names(
        self,
        bindings: Iterable[tuple[str, bool]],
    ) -> tuple[str, ...]:
        """Derive runtime enum members from direct named value bindings."""

        return tuple(
            sorted(
                name
                for name, has_value in bindings
                if has_value
                if not name.startswith("_")
                if self.permits_new_member(name)
            )
        )


PYTHON_ENUM_BASE_AUTHORITY = PythonEnumBaseAuthority()
