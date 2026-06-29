"""Observation graph substrate for fibers, witness groups, and cohorts.

The advisor normalizes collected semantic shapes into structural observations and
then groups them into fibers and witness cohorts. Detectors use these groupings to
reason about partial views, confusability, and coherence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, TypeAlias

from .collection_algebra import sorted_tuple
from .descriptor_algebra import CollectionAttributeProjection
from .export_tools import PublicExportPolicy, derive_public_exports
from .registry_identity import DEFAULT_REGISTRY_KEY_ATTRIBUTE, class_name_registry_key
from metaclass_registry import AutoRegisterMeta

if TYPE_CHECKING:
    from .ast_tools import ParsedModule


class ObservationKind(StrEnum):
    """Canonical observation kinds emitted by the collection substrate."""

    ACCESSOR_WRAPPER = "accessor_wrapper"
    ATTRIBUTE_PROBE = "attribute_probe"
    BUILDER_CALL = "builder_call"
    CLASS_MARKER = "class_marker"
    CONFIG_DISPATCH = "config_dispatch"
    DUAL_AXIS_RESOLUTION = "dual_axis_resolution"
    DYNAMIC_METHOD_INJECTION = "dynamic_method_injection"
    EXPORT_DICT = "export_dict"
    FIELD = "field"
    INTERFACE_GENERATION = "interface_generation"
    LINEAGE_MAPPING = "lineage_mapping"
    LITERAL_DISPATCH = "literal_dispatch"
    METHOD_SHAPE = "method_shape"
    PROJECTION_HELPER = "projection_helper"
    RUNTIME_TYPE_GENERATION = "runtime_type_generation"
    SCOPED_SHAPE_WRAPPER = "scoped_shape_wrapper"
    SENTINEL_TYPE = "sentinel_type"


class StructuralExecutionLevel(StrEnum):
    """Structural execution levels used to group observations."""

    CLASS_BODY = "class_body"
    INIT_BODY = "init_body"
    FUNCTION_BODY = "function_body"
    MODULE_BODY = "module_body"

    @property
    def allows_prefixed_role_field_bundle(self) -> bool:
        return self in {self.CLASS_BODY, self.INIT_BODY}


_ObservationAxis: TypeAlias = tuple[ObservationKind, StructuralExecutionLevel]
_FiberGroupKey: TypeAlias = tuple[ObservationKind, StructuralExecutionLevel, str]


@dataclass(frozen=True)
class _CoherenceCohortCacheKey:
    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    minimum_witnesses: int
    minimum_fibers: int


@dataclass(frozen=True)
class StructuralObservation:
    """Normalized structural fact emitted by a collected shape."""

    file_path: str
    owner_symbol: str
    nominal_witness: str
    line: int
    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    observed_name: str
    fiber_key: str

    @property
    def structural_identity(self) -> tuple[str, int, str]:
        return (self.file_path, self.line, self.owner_symbol)


class StructuralObservationCarrier(ABC, metaclass=AutoRegisterMeta):
    """ABC for objects that can project to a structural observation."""

    __registry_key__ = "OBSERVATION_KIND"
    __skip_if_no_key__ = True

    @property
    @abstractmethod
    def structural_observation(self) -> StructuralObservation:
        raise NotImplementedError


@dataclass(frozen=True)
class ObservationGroup(metaclass=AutoRegisterMeta):
    """Common carrier for grouped observations under one structural axis."""

    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = class_name_registry_key
    __skip_if_no_key__ = True

    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    observations: tuple[StructuralObservation, ...]


@dataclass(frozen=True)
class ObservationFiber(ObservationGroup):
    """All observations that share one observation kind, level, and fiber key."""

    fiber_key: str

    @property
    def observed_name(self) -> str:
        return self.observations[0].observed_name

    @property
    def nominal_witnesses(self) -> tuple[str, ...]:
        return sorted_tuple({item.nominal_witness for item in self.observations})


@dataclass(frozen=True)
class NominalWitnessGroup(ObservationGroup):
    """All observations of one witness under one observation kind and level."""

    nominal_witness: str

    @property
    def observed_names(self) -> tuple[str, ...]:
        return sorted_tuple({item.observed_name for item in self.observations})

    @property
    def fiber_keys(self) -> tuple[str, ...]:
        return sorted_tuple({item.fiber_key for item in self.observations})


_WitnessGroupsByAxis: TypeAlias = dict[
    _ObservationAxis, tuple[NominalWitnessGroup, ...]
]


@dataclass(frozen=True)
class ObservationCohort:
    """Coherent cluster of fibers that share the same witness family."""

    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    nominal_witnesses: tuple[str, ...]
    fibers: tuple[ObservationFiber, ...]
    observed_names = CollectionAttributeProjection[str]("fibers", "observed_name")
    fiber_keys = CollectionAttributeProjection[str]("fibers", "fiber_key")


@dataclass(frozen=True)
class ObservationGraph:
    """Query surface over normalized structural observations."""

    observations: tuple[StructuralObservation, ...]
    _coherence_cohorts_cache: dict[
        _CoherenceCohortCacheKey, tuple[ObservationCohort, ...]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    @cached_property
    def fibers(self) -> tuple[ObservationFiber, ...]:
        grouped: dict[_FiberGroupKey, list[StructuralObservation]] = {}
        for observation in self.observations:
            key = (
                observation.observation_kind,
                observation.execution_level,
                observation.fiber_key,
            )
            grouped.setdefault(key, []).append(observation)
        fibers = [
            ObservationFiber(
                observation_kind=kind,
                execution_level=execution_level,
                fiber_key=fiber_key,
                observations=sorted_tuple(
                    items, key=lambda item: (item.file_path, item.line)
                ),
            )
            for (kind, execution_level, fiber_key), items in grouped.items()
        ]
        return sorted_tuple(
            fibers,
            key=lambda item: (
                item.observation_kind,
                item.execution_level,
                item.fiber_key,
            ),
        )

    @cached_property
    def _fibers_by_axis(self) -> dict[_ObservationAxis, tuple[ObservationFiber, ...]]:
        grouped: dict[_ObservationAxis, list[ObservationFiber]] = {}
        for fiber in self.fibers:
            grouped.setdefault(
                (fiber.observation_kind, fiber.execution_level), []
            ).append(fiber)
        return {axis: tuple(fibers) for axis, fibers in grouped.items()}

    def fibers_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
    ) -> tuple[ObservationFiber, ...]:
        return self._fibers_by_axis.get(
            (observation_kind, execution_level),
            (),
        )

    def fibers_with_min_observations(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
        minimum_observations: int,
    ) -> tuple[ObservationFiber, ...]:
        return tuple(
            (
                fiber
                for fiber in self.fibers_for(observation_kind, execution_level)
                if len(fiber.observations) >= minimum_observations
            )
        )

    def witness_groups_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
    ) -> tuple[NominalWitnessGroup, ...]:
        return self._witness_groups_by_axis.get(
            (observation_kind, execution_level),
            (),
        )

    @cached_property
    def _witness_groups_by_axis(
        self,
    ) -> _WitnessGroupsByAxis:
        grouped_by_axis: dict[
            _ObservationAxis, dict[str, list[StructuralObservation]]
        ] = {}
        for observation in self.observations:
            axis = (observation.observation_kind, observation.execution_level)
            grouped_by_axis.setdefault(axis, {}).setdefault(
                observation.nominal_witness, []
            ).append(observation)

        groups_by_axis: _WitnessGroupsByAxis = {}
        for (observation_kind, execution_level), grouped in grouped_by_axis.items():
            groups = [
                NominalWitnessGroup(
                    observation_kind=observation_kind,
                    execution_level=execution_level,
                    nominal_witness=nominal_witness,
                    observations=sorted_tuple(
                        items,
                        key=lambda item: (
                            item.file_path,
                            item.line,
                            item.owner_symbol,
                        ),
                    ),
                )
                for nominal_witness, items in grouped.items()
            ]
            groups_by_axis[(observation_kind, execution_level)] = sorted_tuple(
                groups, key=lambda item: item.nominal_witness
            )
        return groups_by_axis

    def coherence_cohorts_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
        minimum_witnesses: int = 2,
        minimum_fibers: int = 2,
    ) -> tuple[ObservationCohort, ...]:
        cache_key = _CoherenceCohortCacheKey(
            observation_kind=observation_kind,
            execution_level=execution_level,
            minimum_witnesses=minimum_witnesses,
            minimum_fibers=minimum_fibers,
        )
        cached = self._coherence_cohorts_cache.get(cache_key)
        if cached is not None:
            return cached
        cohorts = self._build_coherence_cohorts_for(
            observation_kind,
            execution_level,
            minimum_witnesses,
            minimum_fibers,
        )
        self._coherence_cohorts_cache[cache_key] = cohorts
        return cohorts

    def _build_coherence_cohorts_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
        minimum_witnesses: int,
        minimum_fibers: int,
    ) -> tuple[ObservationCohort, ...]:
        fibers = self.fibers_for(observation_kind, execution_level)
        relevant_fibers = {
            fiber.fiber_key: fiber
            for fiber in fibers
            if len(fiber.nominal_witnesses) >= minimum_witnesses
        }
        witness_to_fiber_keys = {
            group.nominal_witness: frozenset(
                (
                    fiber_key
                    for fiber_key in group.fiber_keys
                    if fiber_key in relevant_fibers
                )
            )
            for group in self.witness_groups_for(observation_kind, execution_level)
        }
        witness_names = sorted_tuple(
            (
                witness
                for witness, fiber_keys in witness_to_fiber_keys.items()
                if len(fiber_keys) >= minimum_fibers
            )
        )
        cohorts: dict[tuple[tuple[str, ...], tuple[str, ...]], ObservationCohort] = {}
        for left_index, left_name in enumerate(witness_names):
            left_keys = witness_to_fiber_keys[left_name]
            for right_name in witness_names[left_index + 1 :]:
                shared_keys = sorted_tuple(
                    left_keys & witness_to_fiber_keys[right_name]
                )
                if len(shared_keys) < minimum_fibers:
                    continue
                shared_key_set = frozenset(shared_keys)
                supporting_witnesses = sorted_tuple(
                    (
                        witness
                        for witness, fiber_keys in witness_to_fiber_keys.items()
                        if shared_key_set <= fiber_keys
                    )
                )
                if len(supporting_witnesses) < minimum_witnesses:
                    continue
                cohorts[(supporting_witnesses, shared_keys)] = ObservationCohort(
                    observation_kind=observation_kind,
                    execution_level=execution_level,
                    nominal_witnesses=supporting_witnesses,
                    fibers=tuple(
                        (relevant_fibers[fiber_key] for fiber_key in shared_keys)
                    ),
                )
        return sorted_tuple(
            cohorts.values(),
            key=lambda item: (
                item.observation_kind,
                item.execution_level,
                item.nominal_witnesses,
                item.fiber_keys,
            ),
        )


def collect_structural_observations(
    parsed_module: ParsedModule,
) -> tuple[StructuralObservation, ...]:
    """Collect and sort structural observations for one parsed module."""
    return _collect_structural_observations_cached(parsed_module)


@lru_cache(maxsize=None)
def _collect_structural_observations_cached(
    parsed_module: ParsedModule,
) -> tuple[StructuralObservation, ...]:
    from .ast_tools import CollectedFamily, collect_family_items

    observations: list[StructuralObservation] = []
    for family in CollectedFamily.all_registered_families():
        observations.extend(
            (
                item.structural_observation
                for item in collect_family_items(parsed_module, family)
                if isinstance(item, StructuralObservationCarrier)
            )
        )
    return sorted_tuple(
        observations, key=lambda item: (item.file_path, item.line, item.owner_symbol)
    )


def build_observation_graph(modules: list[ParsedModule]) -> ObservationGraph:
    """Build one observation graph from a list of parsed modules."""
    return _build_observation_graph_cached(tuple(modules))


@lru_cache(maxsize=None)
def _build_observation_graph_cached(
    modules: tuple[ParsedModule, ...],
) -> ObservationGraph:
    observations: list[StructuralObservation] = []
    for module in modules:
        observations.extend(collect_structural_observations(module))
    return ObservationGraph(tuple(observations))


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__, types_only=False, allow_callables=True
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
