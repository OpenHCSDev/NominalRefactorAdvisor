"""Proof-carrying finite shape derivation helpers."""

from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
)
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Hashable, TypeVar

from .collection_algebra import sorted_tuple

KeyT = TypeVar("KeyT", bound=Hashable)
RowT = TypeVar("RowT")
ValueT = TypeVar("ValueT")
EnumT = TypeVar("EnumT", bound=Enum)


@dataclass(frozen=True)
class ExhaustivePolicyProof:
    """Runtime proof that a finite policy catalog covers exactly one closed key set."""

    expected_keys: frozenset[Hashable]
    actual_keys: frozenset[Hashable]


@dataclass(frozen=True)
class ProjectionSurfaceCoverageProof:
    """Proof that several derived surfaces cover the same finite semantic axis."""

    surface_names: tuple[str, ...]
    expected_keys: frozenset[Hashable]
    decompression_keys: Mapping[str, str]


@dataclass(frozen=True)
class InjectiveTypeRegistryProof:
    """Proof that one nominal registry axis maps each concrete type to one stable key."""

    key_axis_name: str
    registered_type_names: tuple[str, ...]
    key_names: tuple[str, ...]
    duplicate_key_names: tuple[str, ...]
    duplicate_type_names: tuple[str, ...]
    missing_type_names: tuple[str, ...]
    reverse_lookup_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]

    @property
    def is_injective(self) -> bool:
        return not (
            self.duplicate_key_names
            or self.duplicate_type_names
            or self.missing_type_names
        )

    @classmethod
    def from_key_entries(
        cls,
        *,
        key_axis_name: str,
        key_entries: Iterable[tuple[Hashable, str, tuple[str, ...]]],
        registered_type_names: Iterable[str],
        reverse_lookup_names: Iterable[str] = (),
        consumer_symbols: Iterable[str] = (),
    ) -> "InjectiveTypeRegistryProof":
        """Group proven native keys; source labels are diagnostic, never equality."""
        groups: dict[Hashable, list[tuple[str, tuple[str, ...]]]] = {}
        for key, label, type_names in key_entries:
            groups.setdefault(key, []).append((label, type_names))
        registered = frozenset(registered_type_names)
        key_names: set[str] = set()
        duplicate_keys: set[str] = set()
        key_counts_by_type: Counter[str] = Counter()
        for entries in groups.values():
            labels = {label for label, _ in entries}
            group_types = {name for _, names in entries for name in names}
            key_names.update(labels)
            if len(group_types) > 1:
                duplicate_keys.update(labels)
            key_counts_by_type.update(group_types)
        return cls(
            key_axis_name=key_axis_name,
            registered_type_names=sorted_tuple(registered),
            key_names=sorted_tuple(key_names),
            duplicate_key_names=sorted_tuple(duplicate_keys),
            duplicate_type_names=sorted_tuple(
                {
                    name
                    for name, key_count in key_counts_by_type.items()
                    if key_count > 1
                }
            ),
            missing_type_names=sorted_tuple(registered - key_counts_by_type.keys()),
            reverse_lookup_names=sorted_tuple(set(reverse_lookup_names)),
            consumer_symbols=sorted_tuple(set(consumer_symbols)),
        )

    @classmethod
    def from_partially_resolved_key_entries(
        cls,
        *,
        key_axis_name: str,
        key_entries: Iterable[tuple[Hashable, str, tuple[str, ...]]],
        registered_type_names: Iterable[str],
        unresolved_type_names: Iterable[str],
        reverse_lookup_names: Iterable[str] = (),
        consumer_symbols: Iterable[str] = (),
    ) -> "InjectiveTypeRegistryProof | None":
        """Retain only a decisive counterexample from an incomplete key family.

        Unresolved types cannot establish either injectivity or missing keys. A
        duplicate among the resolved types is nevertheless sufficient to
        disprove injectivity for the complete family.
        """

        registered = frozenset(registered_type_names)
        unresolved = frozenset(unresolved_type_names)
        if not unresolved <= registered:
            raise ValueError("Unresolved registry types must belong to the family")
        proof = cls.from_key_entries(
            key_axis_name=key_axis_name,
            key_entries=key_entries,
            registered_type_names=registered - unresolved,
            reverse_lookup_names=reverse_lookup_names,
            consumer_symbols=consumer_symbols,
        )
        if unresolved and proof.is_injective:
            return None
        return proof

    @classmethod
    def from_type_map(
        cls,
        *,
        key_axis_name: str,
        type_names_by_key: Mapping[str, tuple[str, ...]],
        registered_type_names: Iterable[str],
        reverse_lookup_names: Iterable[str] = (),
        consumer_symbols: Iterable[str] = (),
    ) -> "InjectiveTypeRegistryProof":
        """Use actual string-domain keys, not rendered source expressions."""
        return cls.from_key_entries(
            key_axis_name=key_axis_name,
            key_entries=((key, key, names) for key, names in type_names_by_key.items()),
            registered_type_names=registered_type_names,
            reverse_lookup_names=reverse_lookup_names,
            consumer_symbols=consumer_symbols,
        )


@dataclass(frozen=True)
class ExhaustivePolicyCatalog(Generic[KeyT, RowT]):
    """Typed lookup table that fails unless rows cover exactly one finite key set."""

    rows: tuple[RowT, ...]
    key_of: Callable[[RowT], KeyT]
    expected_keys: frozenset[KeyT]

    @classmethod
    def for_enum(
        cls,
        enum_type: type[EnumT],
        rows: Iterable[RowT],
        key_of: Callable[[RowT], EnumT],
    ) -> "ExhaustivePolicyCatalog[EnumT, RowT]":
        return cls(tuple(rows), key_of, frozenset(enum_type))

    def __post_init__(self) -> None:
        duplicate_keys = self.duplicate_keys
        if duplicate_keys:
            keys = ", ".join((repr(key) for key in duplicate_keys))
            raise ValueError(f"duplicate finite policy keys: {keys}")
        if self.actual_keys != self.expected_keys:
            missing = sorted_tuple(self.expected_keys - self.actual_keys, key=repr)
            unexpected = sorted_tuple(self.actual_keys - self.expected_keys, key=repr)
            raise ValueError(
                "finite policy catalog coverage mismatch: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

    @property
    def actual_keys(self) -> frozenset[KeyT]:
        return frozenset((self.key_of(row) for row in self.rows))

    @property
    def duplicate_keys(self) -> tuple[KeyT, ...]:
        seen: set[KeyT] = set()
        duplicates: set[KeyT] = set()
        for row in self.rows:
            key = self.key_of(row)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        return sorted_tuple(duplicates, key=repr)

    @property
    def proof(self) -> ExhaustivePolicyProof:
        return ExhaustivePolicyProof(
            expected_keys=frozenset(self.expected_keys),
            actual_keys=frozenset(self.actual_keys),
        )

    @property
    def by_key(self) -> Mapping[KeyT, RowT]:
        return {self.key_of(row): row for row in self.rows}

    def lookup(self, key: KeyT) -> RowT:
        return self.by_key[key]

    def project(self, projector: Callable[[RowT], ValueT]) -> Mapping[KeyT, ValueT]:
        return {key: projector(row) for key, row in self.by_key.items()}


@dataclass(frozen=True)
class ProjectionSurfaceCatalog(Generic[KeyT, RowT]):
    """Proof that several projection surfaces derive from one closed key axis."""

    rows: tuple[RowT, ...]
    surface_of: Callable[[RowT], str]
    key_of: Callable[[RowT], KeyT]
    expected_keys: frozenset[KeyT]
    decompression_key_of: Callable[[str], str]

    def __post_init__(self) -> None:
        if not self.surface_names:
            raise ValueError(
                "projection surface catalog must contain at least one surface"
            )
        for surface_name in self.surface_names:
            decompression_key = self.decompression_key_of(surface_name)
            if not decompression_key:
                raise ValueError(
                    f"projection surface {surface_name!r} lacks a decompression key"
                )
            duplicate_keys = self.duplicate_keys_for_surface(surface_name)
            if duplicate_keys:
                keys = ", ".join((repr(key) for key in duplicate_keys))
                raise ValueError(
                    f"projection surface {surface_name!r} has duplicate keys: {keys}"
                )
            actual_keys = self.keys_for_surface(surface_name)
            if actual_keys != self.expected_keys:
                missing = sorted_tuple(self.expected_keys - actual_keys, key=repr)
                unexpected = sorted_tuple(actual_keys - self.expected_keys, key=repr)
                raise ValueError(
                    "projection surface coverage mismatch: "
                    f"surface={surface_name!r}, missing={missing!r}, "
                    f"unexpected={unexpected!r}"
                )

    @classmethod
    def for_enum(
        cls,
        enum_type: type[EnumT],
        rows: Iterable[RowT],
        *,
        surface_of: Callable[[RowT], str],
        key_of: Callable[[RowT], EnumT],
        decompression_key_of: Callable[[str], str],
    ) -> "ProjectionSurfaceCatalog[EnumT, RowT]":
        return cls(
            tuple(rows),
            surface_of,
            key_of,
            frozenset(enum_type),
            decompression_key_of,
        )

    @property
    def rows_by_surface(self) -> Mapping[str, tuple[RowT, ...]]:
        grouped: dict[str, list[RowT]] = defaultdict(list)
        for row in self.rows:
            grouped[self.surface_of(row)].append(row)
        return {
            surface_name: tuple(rows)
            for surface_name, rows in sorted(grouped.items(), key=lambda item: item[0])
        }

    @property
    def surface_names(self) -> tuple[str, ...]:
        return sorted_tuple(frozenset((self.surface_of(row) for row in self.rows)))

    def keys_for_surface(self, surface_name: str) -> frozenset[KeyT]:
        return frozenset(
            (
                self.key_of(row)
                for row in self.rows
                if self.surface_of(row) == surface_name
            )
        )

    def duplicate_keys_for_surface(self, surface_name: str) -> tuple[KeyT, ...]:
        seen: set[KeyT] = set()
        duplicates: set[KeyT] = set()
        for row in self.rows:
            if self.surface_of(row) != surface_name:
                continue
            key = self.key_of(row)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        return sorted_tuple(duplicates, key=repr)

    @property
    def proof(self) -> ProjectionSurfaceCoverageProof:
        return ProjectionSurfaceCoverageProof(
            surface_names=self.surface_names,
            expected_keys=frozenset(self.expected_keys),
            decompression_keys={
                surface_name: self.decompression_key_of(surface_name)
                for surface_name in self.surface_names
            },
        )
