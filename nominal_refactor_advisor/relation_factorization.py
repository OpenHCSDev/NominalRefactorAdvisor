"""Exact finite certificates for required-relation factorization.

The module factors a closed, explicitly-carried finite relation.  It never
infers semantic authority: callers supply the admitted implementation and
consumer carriers and the exact required pairs.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from itertools import combinations
from typing import Any, ClassVar

from .collection_algebra import sorted_tuple
from .factorization import FactorizationRow, FormalConceptLattice

CERTIFICATE_SCHEMA = "relation-factorization-certificate/v1"
LEXICAL_TIE_BREAK = "maximum-cardinality-then-lexical/v1"
PARENT_TIE_BREAK = "scope-cardinality-scope-lexical-key/v1"
COVER_TIE_BREAK = "minimum-cardinality-then-lexical/v1"


def _validated_name(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be nonblank and trimmed")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must be NFC-normalized")
    return value


def _canonical_names(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    raw = tuple(values)
    normalized = tuple(_validated_name(value, field_name) for value in raw)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} contains duplicate values")
    return sorted_tuple(normalized)


@dataclass(frozen=True, order=True)
class RequiredRelationPair:
    """One exact required relation row."""

    implementation_key: str
    consumer: str

    def __post_init__(self) -> None:
        _validated_name(self.implementation_key, "implementation_key")
        _validated_name(self.consumer, "consumer")


@dataclass(frozen=True, init=False)
class RequiredRelation:
    """A finite relation with explicit zero-degree endpoint carriers."""

    implementation_keys: tuple[str, ...]
    consumers: tuple[str, ...]
    pairs: tuple[RequiredRelationPair, ...]

    def __init__(
        self,
        implementation_keys: Iterable[str],
        consumers: Iterable[str],
        pairs: Iterable[RequiredRelationPair],
    ) -> None:
        implementation_tuple = _canonical_names(
            implementation_keys, "implementation_keys"
        )
        consumer_tuple = _canonical_names(consumers, "consumers")
        raw_pairs = tuple(pairs)
        if any(not isinstance(pair, RequiredRelationPair) for pair in raw_pairs):
            raise TypeError("pairs must contain RequiredRelationPair values")
        if len(set(raw_pairs)) != len(raw_pairs):
            raise ValueError("pairs contains duplicate values")
        implementation_set = frozenset(implementation_tuple)
        consumer_set = frozenset(consumer_tuple)
        for pair in raw_pairs:
            if pair.implementation_key not in implementation_set:
                raise ValueError(
                    f"pair has foreign implementation_key {pair.implementation_key!r}"
                )
            if pair.consumer not in consumer_set:
                raise ValueError(f"pair has foreign consumer {pair.consumer!r}")
        object.__setattr__(self, "implementation_keys", implementation_tuple)
        object.__setattr__(self, "consumers", consumer_tuple)
        object.__setattr__(self, "pairs", sorted_tuple(raw_pairs))

    @classmethod
    def from_pairs(
        cls,
        implementation_keys: Iterable[str],
        consumers: Iterable[str],
        pairs: Iterable[tuple[str, str] | RequiredRelationPair],
    ) -> RequiredRelation:
        return cls(
            implementation_keys,
            consumers,
            (
                pair
                if isinstance(pair, RequiredRelationPair)
                else RequiredRelationPair(*pair)
                for pair in pairs
            ),
        )

    def consumers_for(self, implementation_key: str) -> frozenset[str]:
        if implementation_key not in self.implementation_keys:
            raise KeyError(implementation_key)
        return frozenset(
            pair.consumer
            for pair in self.pairs
            if pair.implementation_key == implementation_key
        )

    def implementations_for(self, consumer: str) -> frozenset[str]:
        if consumer not in self.consumers:
            raise KeyError(consumer)
        return frozenset(
            pair.implementation_key for pair in self.pairs if pair.consumer == consumer
        )

    def with_pairs(self, pairs: Iterable[RequiredRelationPair]) -> RequiredRelation:
        return RequiredRelation(self.implementation_keys, self.consumers, pairs)

    def difference(self, other: RequiredRelation) -> RequiredRelation:
        if (
            self.implementation_keys != other.implementation_keys
            or self.consumers != other.consumers
        ):
            raise ValueError("relation carriers differ")
        return self.with_pairs(frozenset(self.pairs) - frozenset(other.pairs))


@dataclass(frozen=True)
class ExactSearchLimits:
    """Caller-owned finite ceilings for every exponential search phase."""

    max_pairs: int = 16
    max_implementation_keys: int = 16
    max_consumers: int = 16
    max_laminar_subsets: int = 65_536
    max_fca_intent_subsets: int = 65_536
    max_provider_candidates: int = 128
    max_cover_combinations: int = 250_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


_DEFAULT_EXACT_SEARCH_LIMITS = ExactSearchLimits()


class ExactSearchLimitExceeded(RuntimeError):
    """One exact phase exceeded a declared finite ceiling."""

    def __init__(self, phase: str, required: int, limit: int) -> None:
        self.phase = phase
        self.required = required
        self.limit = limit
        super().__init__(f"{phase} requires {required} states; limit is {limit}")


def _bounded(phase: str, required: int, limit: int) -> None:
    if required > limit:
        raise ExactSearchLimitExceeded(phase, required, limit)


def _validate_relation_limits(
    relation: RequiredRelation, limits: ExactSearchLimits
) -> None:
    _bounded("relation_pairs", len(relation.pairs), limits.max_pairs)
    _bounded(
        "implementation_carrier",
        len(relation.implementation_keys),
        limits.max_implementation_keys,
    )
    _bounded("consumer_carrier", len(relation.consumers), limits.max_consumers)


def _is_laminar(relation: RequiredRelation) -> bool:
    scopes = tuple(relation.consumers_for(key) for key in relation.implementation_keys)
    for left, right in combinations(scopes, 2):
        if left & right and not (left <= right or right <= left):
            return False
    return True


def _expected_parent_maps(
    retained: RequiredRelation,
) -> tuple[tuple[tuple[str, str | None], ...], tuple[tuple[str, str | None], ...]]:
    scopes = {key: retained.consumers_for(key) for key in retained.implementation_keys}
    ordered_keys = sorted_tuple(
        retained.implementation_keys,
        key=lambda key: (len(scopes[key]), sorted_tuple(scopes[key]), key),
    )
    implementation_parents: list[tuple[str, str | None]] = []
    for index, child in enumerate(ordered_keys):
        parent = next(
            (
                candidate
                for candidate in ordered_keys[index + 1 :]
                if scopes[child] <= scopes[candidate]
            ),
            None,
        )
        implementation_parents.append((child, parent))
    consumer_parents = tuple(
        (
            consumer,
            next((key for key in ordered_keys if consumer in scopes[key]), None),
        )
        for consumer in retained.consumers
    )
    return (
        sorted_tuple(implementation_parents),
        sorted_tuple(consumer_parents),
    )


def _descendant_consumers(
    implementation_key: str,
    implementation_parents: Mapping[str, str | None],
    consumer_parents: Mapping[str, str | None],
) -> frozenset[str]:
    def descends_from(candidate: str, ancestor: str) -> bool:
        seen: set[str] = set()
        current: str | None = candidate
        while current is not None:
            if current == ancestor:
                return True
            if current in seen:
                raise ValueError("implementation parent map contains a cycle")
            seen.add(current)
            current = implementation_parents[current]
        return False

    return frozenset(
        consumer
        for consumer, parent in consumer_parents.items()
        if parent is not None and descends_from(parent, implementation_key)
    )


@dataclass(frozen=True)
class SingleParentArrangement:
    """Canonical single-parent witness validated against one retained relation.

    ``retained`` is construction-only validation context.  It is deliberately
    not stored or serialized, so the arrangement cannot become a second
    writable relation authority.
    """

    implementation_parents: tuple[tuple[str, str | None], ...]
    consumer_parents: tuple[tuple[str, str | None], ...]
    tie_break: str
    retained: InitVar[RequiredRelation]

    def __post_init__(self, retained: RequiredRelation) -> None:
        if not isinstance(retained, RequiredRelation):
            raise TypeError("retained validation context must be RequiredRelation")
        if self.tie_break != PARENT_TIE_BREAK:
            raise ValueError("unsupported parent tie-break policy")
        if type(self.implementation_parents) is not tuple:
            raise TypeError("implementation_parents must be an exact tuple")
        if type(self.consumer_parents) is not tuple:
            raise TypeError("consumer_parents must be an exact tuple")
        implementation_rows = self.implementation_parents
        consumer_rows = self.consumer_parents
        if any(
            type(row) is not tuple or len(row) != 2
            for row in (*implementation_rows, *consumer_rows)
        ):
            raise TypeError("parent maps must contain two-item tuple rows")
        implementation_children = tuple(row[0] for row in implementation_rows)
        consumer_children = tuple(row[0] for row in consumer_rows)
        if len(set(implementation_children)) != len(implementation_children):
            raise ValueError("duplicate implementation parent-map child")
        if len(set(consumer_children)) != len(consumer_children):
            raise ValueError("duplicate consumer parent-map child")
        if implementation_rows != sorted_tuple(implementation_rows):
            raise ValueError("implementation parent map must be canonical")
        if consumer_rows != sorted_tuple(consumer_rows):
            raise ValueError("consumer parent map must be canonical")
        if implementation_children != retained.implementation_keys:
            raise ValueError("implementation parent map must cover exact carrier")
        if consumer_children != retained.consumers:
            raise ValueError("consumer parent map must cover exact carrier")
        implementation_set = frozenset(retained.implementation_keys)
        for child, parent in implementation_rows:
            if parent is not None and parent not in implementation_set:
                raise ValueError(f"foreign implementation parent {parent!r}")
            if parent == child:
                raise ValueError("implementation cannot parent itself")
        for _, parent in consumer_rows:
            if parent is not None and parent not in implementation_set:
                raise ValueError(f"foreign consumer parent {parent!r}")
        implementation_map = dict(implementation_rows)
        consumer_map = dict(consumer_rows)
        for key in retained.implementation_keys:
            seen: set[str] = set()
            current: str | None = key
            while current is not None:
                if current in seen:
                    raise ValueError("implementation parent map contains a cycle")
                seen.add(current)
                current = implementation_map[current]
        expected_implementations, expected_consumers = _expected_parent_maps(retained)
        if implementation_rows != expected_implementations:
            raise ValueError("implementation parent map violates canonical policy")
        if consumer_rows != expected_consumers:
            raise ValueError("consumer parent map violates canonical policy")
        for key in retained.implementation_keys:
            if _descendant_consumers(key, implementation_map, consumer_map) != (
                retained.consumers_for(key)
            ):
                raise ValueError("arrangement descendants do not equal retained scope")

    @classmethod
    def from_relation(cls, retained: RequiredRelation) -> SingleParentArrangement:
        if not _is_laminar(retained):
            raise ValueError("single-parent arrangement requires a laminar relation")
        implementation_parents, consumer_parents = _expected_parent_maps(retained)
        return cls(
            implementation_parents=implementation_parents,
            consumer_parents=consumer_parents,
            tie_break=PARENT_TIE_BREAK,
            retained=retained,
        )

    def descendants_for(self, implementation_key: str) -> frozenset[str]:
        implementation_map = dict(self.implementation_parents)
        if implementation_key not in implementation_map:
            raise KeyError(implementation_key)
        return _descendant_consumers(
            implementation_key,
            implementation_map,
            dict(self.consumer_parents),
        )


def _validate_exact_count(value: int, field_name: str, *, positive: bool) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    minimum = 1 if positive else 0
    if value < minimum:
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be {bound}")


@dataclass(frozen=True)
class LaminarSearchStatistics:
    subsets_examined: int
    feasible_laminar_subsets: int
    optimal_subsets: int

    def __post_init__(self) -> None:
        _validate_exact_count(self.subsets_examined, "subsets_examined", positive=True)
        _validate_exact_count(
            self.feasible_laminar_subsets,
            "feasible_laminar_subsets",
            positive=True,
        )
        _validate_exact_count(self.optimal_subsets, "optimal_subsets", positive=True)
        if self.feasible_laminar_subsets > self.subsets_examined:
            raise ValueError("feasible_laminar_subsets exceeds subsets_examined")
        if self.optimal_subsets > self.feasible_laminar_subsets:
            raise ValueError("optimal_subsets exceeds feasible_laminar_subsets")


@dataclass(frozen=True)
class LaminarRepairCertificate:
    required: RequiredRelation
    retained: RequiredRelation
    residual: RequiredRelation
    ancestry_gap: tuple[RequiredRelationPair, ...]
    arrangement: SingleParentArrangement
    statistics: LaminarSearchStatistics
    search_limits: ExactSearchLimits
    tie_break: str = LEXICAL_TIE_BREAK

    @property
    def optimal_repair_count(self) -> int:
        return self.statistics.optimal_subsets

    def verify(self) -> None:
        expected = factor_laminar_relation(self.required, limits=self.search_limits)
        if self != expected:
            raise ValueError("laminar repair certificate does not recompute exactly")


def factor_laminar_relation(
    relation: RequiredRelation,
    *,
    limits: ExactSearchLimits = _DEFAULT_EXACT_SEARCH_LIMITS,
) -> LaminarRepairCertificate:
    """Return the exact maximum-cardinality deletion-only laminar repair."""

    _validate_relation_limits(relation, limits)
    state_count = 1 << len(relation.pairs)
    _bounded("laminar_subsets", state_count, limits.max_laminar_subsets)
    feasible: list[RequiredRelation] = []
    best_size = -1
    optimal: list[RequiredRelation] = []
    for mask in range(state_count):
        candidate = relation.with_pairs(
            pair for index, pair in enumerate(relation.pairs) if mask & (1 << index)
        )
        if not _is_laminar(candidate):
            continue
        feasible.append(candidate)
        candidate_size = len(candidate.pairs)
        if candidate_size > best_size:
            best_size = candidate_size
            optimal = [candidate]
        elif candidate_size == best_size:
            optimal.append(candidate)
    optimal = sorted(
        optimal,
        key=lambda candidate: tuple(
            (pair.implementation_key, pair.consumer) for pair in candidate.pairs
        ),
    )
    retained = optimal[0]
    residual = relation.difference(retained)
    return LaminarRepairCertificate(
        required=relation,
        retained=retained,
        residual=residual,
        ancestry_gap=residual.pairs,
        arrangement=SingleParentArrangement.from_relation(retained),
        statistics=LaminarSearchStatistics(
            subsets_examined=state_count,
            feasible_laminar_subsets=len(feasible),
            optimal_subsets=len(optimal),
        ),
        search_limits=limits,
    )


@dataclass(frozen=True, order=True)
class ProviderRectangle:
    """One nonempty maximal biclique in the residual relation."""

    implementation_keys: tuple[str, ...]
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.implementation_keys or not self.consumers:
            raise ValueError("provider rectangles must be nonempty")
        if self.implementation_keys != _canonical_names(
            self.implementation_keys, "implementation_keys"
        ):
            raise ValueError("implementation_keys must be canonical")
        if self.consumers != _canonical_names(self.consumers, "consumers"):
            raise ValueError("consumers must be canonical")

    @property
    def pairs(self) -> frozenset[RequiredRelationPair]:
        return frozenset(
            RequiredRelationPair(implementation_key, consumer)
            for implementation_key in self.implementation_keys
            for consumer in self.consumers
        )


@dataclass(frozen=True)
class FormalConceptSearchStatistics:
    intent_subsets_examined: int
    maximal_biclique_candidates: int

    def __post_init__(self) -> None:
        _validate_exact_count(
            self.intent_subsets_examined,
            "intent_subsets_examined",
            positive=True,
        )
        _validate_exact_count(
            self.maximal_biclique_candidates,
            "maximal_biclique_candidates",
            positive=False,
        )
        if self.maximal_biclique_candidates > self.intent_subsets_examined:
            raise ValueError(
                "maximal_biclique_candidates exceeds intent_subsets_examined"
            )


@dataclass(frozen=True)
class ProviderCoverSearchStatistics:
    combinations_examined: int
    optimal_covers: int

    def __post_init__(self) -> None:
        _validate_exact_count(
            self.combinations_examined, "combinations_examined", positive=False
        )
        _validate_exact_count(self.optimal_covers, "optimal_covers", positive=True)
        if self.combinations_examined == 0 and self.optimal_covers != 1:
            raise ValueError("zero combinations requires one empty optimal cover")
        if (
            self.combinations_examined > 0
            and self.optimal_covers > self.combinations_examined
        ):
            raise ValueError("optimal_covers exceeds combinations_examined")


@dataclass(frozen=True)
class ProviderCoverCertificate:
    relation: RequiredRelation
    rectangles: tuple[ProviderRectangle, ...]
    candidate_rectangles: tuple[ProviderRectangle, ...]
    formal_concept_statistics: FormalConceptSearchStatistics
    cover_statistics: ProviderCoverSearchStatistics
    search_limits: ExactSearchLimits
    tie_break: str = COVER_TIE_BREAK

    @property
    def optimal_cover_count(self) -> int:
        return self.cover_statistics.optimal_covers

    def verify(self) -> None:
        expected = factor_provider_cover(self.relation, limits=self.search_limits)
        if self != expected:
            raise ValueError("provider cover certificate does not recompute exactly")


def _provider_candidates(
    relation: RequiredRelation, limits: ExactSearchLimits
) -> tuple[tuple[ProviderRectangle, ...], FormalConceptSearchStatistics]:
    observed_consumers = sorted_tuple(
        frozenset(pair.consumer for pair in relation.pairs)
    )
    intent_count = 1 << len(observed_consumers)
    _bounded("fca_intent_subsets", intent_count, limits.max_fca_intent_subsets)
    rows = tuple(
        FactorizationRow.from_mapping(
            implementation_key,
            {consumer: True for consumer in relation.consumers_for(implementation_key)},
        )
        for implementation_key in relation.implementation_keys
    )
    lattice = FormalConceptLattice.from_rows(rows)
    candidates = {
        ProviderRectangle(
            implementation_keys=sorted_tuple(concept.extent),
            consumers=sorted_tuple(
                attribute_name
                for attribute_name, attribute_value in concept.intent
                if attribute_value is True
            ),
        )
        for concept in lattice.concepts
        if concept.extent
        and any(attribute_value is True for _, attribute_value in concept.intent)
    }
    exact_candidates = tuple(
        rectangle
        for rectangle in sorted_tuple(candidates)
        if rectangle.pairs <= frozenset(relation.pairs)
    )
    _bounded(
        "provider_candidates", len(exact_candidates), limits.max_provider_candidates
    )
    return exact_candidates, FormalConceptSearchStatistics(
        intent_subsets_examined=intent_count,
        maximal_biclique_candidates=len(exact_candidates),
    )


def factor_provider_cover(
    relation: RequiredRelation,
    *,
    limits: ExactSearchLimits = _DEFAULT_EXACT_SEARCH_LIMITS,
) -> ProviderCoverCertificate:
    """Return the exact minimum cover by maximal residual bicliques."""

    _validate_relation_limits(relation, limits)
    candidates, formal_statistics = _provider_candidates(relation, limits)
    target = frozenset(relation.pairs)
    if not target:
        return ProviderCoverCertificate(
            relation=relation,
            rectangles=(),
            candidate_rectangles=candidates,
            formal_concept_statistics=formal_statistics,
            cover_statistics=ProviderCoverSearchStatistics(0, 1),
            search_limits=limits,
        )
    examined = 0
    optimum: list[tuple[ProviderRectangle, ...]] = []
    for size in range(1, len(candidates) + 1):
        for candidate_cover in combinations(candidates, size):
            examined += 1
            _bounded(
                "provider_cover_combinations",
                examined,
                limits.max_cover_combinations,
            )
            union = frozenset(
                pair for rectangle in candidate_cover for pair in rectangle.pairs
            )
            if union == target:
                optimum.append(tuple(candidate_cover))
        if optimum:
            break
    if not optimum:
        raise ValueError("maximal biclique candidates do not cover residual relation")
    optimum = sorted(optimum)
    return ProviderCoverCertificate(
        relation=relation,
        rectangles=optimum[0],
        candidate_rectangles=candidates,
        formal_concept_statistics=formal_statistics,
        cover_statistics=ProviderCoverSearchStatistics(examined, len(optimum)),
        search_limits=limits,
    )


@dataclass(frozen=True)
class OwnerInvalidationSurface:
    implementation_key: str
    derived_consumers: tuple[str, ...]
    residual_pairs: tuple[RequiredRelationPair, ...]
    is_choke_point: bool

    def __post_init__(self) -> None:
        _validated_name(self.implementation_key, "implementation_key")
        if self.derived_consumers != _canonical_names(
            self.derived_consumers, "derived_consumers"
        ):
            raise ValueError("derived_consumers must be canonical")
        if any(type(pair) is not RequiredRelationPair for pair in self.residual_pairs):
            raise TypeError(
                "residual_pairs must contain exact RequiredRelationPair values"
            )
        if len(set(self.residual_pairs)) != len(self.residual_pairs):
            raise ValueError("residual_pairs contains duplicate values")
        if any(
            pair.implementation_key != self.implementation_key
            for pair in self.residual_pairs
        ):
            raise ValueError("residual pair implementation_key differs from surface")
        if self.residual_pairs != sorted_tuple(self.residual_pairs):
            raise ValueError("residual_pairs must be canonical")
        expected = bool(self.derived_consumers) and not self.residual_pairs
        if self.is_choke_point is not expected:
            raise ValueError("is_choke_point must be derived from the surface")


@dataclass(frozen=True)
class RelationFactorizationCertificate:
    """One atomic, self-recomputing finite factorization certificate."""

    SCHEMA: ClassVar[str] = CERTIFICATE_SCHEMA

    relation: RequiredRelation
    laminar_repair: LaminarRepairCertificate
    provider_cover: ProviderCoverCertificate
    invalidation_surfaces: tuple[OwnerInvalidationSurface, ...]
    search_limits: ExactSearchLimits
    schema: str = CERTIFICATE_SCHEMA

    @classmethod
    def factor(
        cls,
        relation: RequiredRelation,
        *,
        limits: ExactSearchLimits = _DEFAULT_EXACT_SEARCH_LIMITS,
    ) -> RelationFactorizationCertificate:
        repair = factor_laminar_relation(relation, limits=limits)
        cover = factor_provider_cover(repair.residual, limits=limits)
        surfaces = tuple(
            OwnerInvalidationSurface(
                implementation_key=key,
                derived_consumers=sorted_tuple(repair.retained.consumers_for(key)),
                residual_pairs=tuple(
                    pair
                    for pair in repair.residual.pairs
                    if pair.implementation_key == key
                ),
                is_choke_point=bool(repair.retained.consumers_for(key))
                and not any(
                    pair.implementation_key == key for pair in repair.residual.pairs
                ),
            )
            for key in relation.implementation_keys
        )
        return cls(
            relation=relation,
            laminar_repair=repair,
            provider_cover=cover,
            invalidation_surfaces=surfaces,
            search_limits=limits,
        )

    def verify(self) -> None:
        if self.schema != CERTIFICATE_SCHEMA:
            raise ValueError("unsupported relation factorization schema")
        expected = type(self).factor(self.relation, limits=self.search_limits)
        if self != expected:
            raise ValueError("relation factorization certificate does not recompute")

    def to_dict(self) -> dict[str, Any]:
        def pair_rows(relation: RequiredRelation) -> list[list[str]]:
            return [[pair.implementation_key, pair.consumer] for pair in relation.pairs]

        def relation_row(relation: RequiredRelation) -> dict[str, Any]:
            return {
                "implementation_keys": list(relation.implementation_keys),
                "consumers": list(relation.consumers),
                "pairs": pair_rows(relation),
            }

        repair = self.laminar_repair
        cover = self.provider_cover
        return {
            "schema": self.schema,
            "relation": relation_row(self.relation),
            "search_limits": dict(sorted(self.search_limits.__dict__.items())),
            "laminar_repair": {
                "retained": relation_row(repair.retained),
                "residual": relation_row(repair.residual),
                "ancestry_gap": [
                    [pair.implementation_key, pair.consumer]
                    for pair in repair.ancestry_gap
                ],
                "arrangement": {
                    "implementation_parents": [
                        [child, parent]
                        for child, parent in repair.arrangement.implementation_parents
                    ],
                    "consumer_parents": [
                        [child, parent]
                        for child, parent in repair.arrangement.consumer_parents
                    ],
                    "tie_break": repair.arrangement.tie_break,
                },
                "statistics": {
                    "subsets_examined": repair.statistics.subsets_examined,
                    "feasible_laminar_subsets": (
                        repair.statistics.feasible_laminar_subsets
                    ),
                    "optimal_subsets": repair.statistics.optimal_subsets,
                },
                "tie_break": repair.tie_break,
            },
            "provider_cover": {
                "relation": relation_row(cover.relation),
                "rectangles": [
                    [list(rectangle.implementation_keys), list(rectangle.consumers)]
                    for rectangle in cover.rectangles
                ],
                "candidate_rectangles": [
                    [list(rectangle.implementation_keys), list(rectangle.consumers)]
                    for rectangle in cover.candidate_rectangles
                ],
                "formal_concept_statistics": {
                    "intent_subsets_examined": (
                        cover.formal_concept_statistics.intent_subsets_examined
                    ),
                    "maximal_biclique_candidates": (
                        cover.formal_concept_statistics.maximal_biclique_candidates
                    ),
                },
                "cover_statistics": {
                    "combinations_examined": (
                        cover.cover_statistics.combinations_examined
                    ),
                    "optimal_covers": cover.cover_statistics.optimal_covers,
                },
                "tie_break": cover.tie_break,
            },
            "invalidation_surfaces": [
                {
                    "implementation_key": surface.implementation_key,
                    "derived_consumers": list(surface.derived_consumers),
                    "residual_pairs": [
                        [pair.implementation_key, pair.consumer]
                        for pair in surface.residual_pairs
                    ],
                    "is_choke_point": surface.is_choke_point,
                }
                for surface in self.invalidation_surfaces
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def content_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
