"""Observation graph substrate for fibers, witness groups, and cohorts.

The advisor normalizes collected semantic shapes into structural observations and
then groups them into fibers and witness cohorts. Detectors use these groupings to
reason about partial views, confusability, and coherence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

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


class StructuralObservationCarrier(ABC):
    """Protocol for objects that can project to a structural observation."""

    @property
    @abstractmethod
    def structural_observation(self) -> StructuralObservation:
        raise NotImplementedError


@dataclass(frozen=True)
class ObservationFiber:
    """All observations that share one observation kind, level, and fiber key."""

    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    fiber_key: str
    observations: tuple[StructuralObservation, ...]

    @property
    def observed_name(self) -> str:
        return self.observations[0].observed_name

    @property
    def nominal_witnesses(self) -> tuple[str, ...]:
        return tuple(sorted({item.nominal_witness for item in self.observations}))


@dataclass(frozen=True)
class NominalWitnessGroup:
    """All observations of one witness under one observation kind and level."""

    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    nominal_witness: str
    observations: tuple[StructuralObservation, ...]

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(sorted({item.observed_name for item in self.observations}))

    @property
    def fiber_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.fiber_key for item in self.observations}))


@dataclass(frozen=True)
class ObservationCohort:
    """Coherent cluster of fibers that share the same witness family."""

    observation_kind: ObservationKind
    execution_level: StructuralExecutionLevel
    nominal_witnesses: tuple[str, ...]
    fibers: tuple[ObservationFiber, ...]

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(fiber.observed_name for fiber in self.fibers)

    @property
    def fiber_keys(self) -> tuple[str, ...]:
        return tuple(fiber.fiber_key for fiber in self.fibers)


@dataclass(frozen=True)
class ObservationGraph:
    """Query surface over normalized structural observations."""

    observations: tuple[StructuralObservation, ...]

    @property
    def fibers(self) -> tuple[ObservationFiber, ...]:
        grouped: dict[
            tuple[ObservationKind, StructuralExecutionLevel, str],
            list[StructuralObservation],
        ] = {}
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
                observations=tuple(
                    sorted(items, key=lambda item: (item.file_path, item.line))
                ),
            )
            for (kind, execution_level, fiber_key), items in grouped.items()
        ]
        return tuple(
            sorted(
                fibers,
                key=lambda item: (
                    item.observation_kind,
                    item.execution_level,
                    item.fiber_key,
                ),
            )
        )

    def fibers_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
    ) -> tuple[ObservationFiber, ...]:
        return tuple(
            fiber
            for fiber in self.fibers
            if fiber.observation_kind == observation_kind
            and fiber.execution_level == execution_level
        )

    def fibers_with_min_observations(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
        minimum_observations: int,
    ) -> tuple[ObservationFiber, ...]:
        return tuple(
            fiber
            for fiber in self.fibers_for(observation_kind, execution_level)
            if len(fiber.observations) >= minimum_observations
        )

    def witness_groups_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
    ) -> tuple[NominalWitnessGroup, ...]:
        grouped: dict[str, list[StructuralObservation]] = {}
        for observation in self.observations:
            if observation.observation_kind != observation_kind:
                continue
            if observation.execution_level != execution_level:
                continue
            grouped.setdefault(observation.nominal_witness, []).append(observation)
        groups = [
            NominalWitnessGroup(
                observation_kind=observation_kind,
                execution_level=execution_level,
                nominal_witness=nominal_witness,
                observations=tuple(
                    sorted(
                        items,
                        key=lambda item: (
                            item.file_path,
                            item.line,
                            item.owner_symbol,
                        ),
                    )
                ),
            )
            for nominal_witness, items in grouped.items()
        ]
        return tuple(sorted(groups, key=lambda item: item.nominal_witness))

    def coherence_cohorts_for(
        self,
        observation_kind: ObservationKind,
        execution_level: StructuralExecutionLevel,
        minimum_witnesses: int = 2,
        minimum_fibers: int = 2,
    ) -> tuple[ObservationCohort, ...]:
        fibers = self.fibers_for(observation_kind, execution_level)
        relevant_fibers = {
            fiber.fiber_key: fiber
            for fiber in fibers
            if len(fiber.nominal_witnesses) >= minimum_witnesses
        }
        witness_to_fiber_keys = {
            group.nominal_witness: frozenset(
                fiber_key
                for fiber_key in group.fiber_keys
                if fiber_key in relevant_fibers
            )
            for group in self.witness_groups_for(observation_kind, execution_level)
        }
        witness_names = tuple(
            sorted(
                witness
                for witness, fiber_keys in witness_to_fiber_keys.items()
                if len(fiber_keys) >= minimum_fibers
            )
        )
        cohorts: dict[tuple[tuple[str, ...], tuple[str, ...]], ObservationCohort] = {}
        for left_index, left_name in enumerate(witness_names):
            left_keys = witness_to_fiber_keys[left_name]
            for right_name in witness_names[left_index + 1 :]:
                shared_keys = tuple(
                    sorted(left_keys & witness_to_fiber_keys[right_name])
                )
                if len(shared_keys) < minimum_fibers:
                    continue
                shared_key_set = frozenset(shared_keys)
                supporting_witnesses = tuple(
                    sorted(
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
                        relevant_fibers[fiber_key] for fiber_key in shared_keys
                    ),
                )
        return tuple(
            sorted(
                cohorts.values(),
                key=lambda item: (
                    item.observation_kind,
                    item.execution_level,
                    item.nominal_witnesses,
                    item.fiber_keys,
                ),
            )
        )


def collect_structural_observations(
    parsed_module: ParsedModule,
) -> tuple[StructuralObservation, ...]:
    """Collect and sort structural observations for one parsed module."""
    from .ast_tools import CollectedFamily, collect_family_items

    observations: list[StructuralObservation] = []
    for family in CollectedFamily.all_registered_families():
        observations.extend(
            item.structural_observation
            for item in collect_family_items(parsed_module, family)
            if isinstance(item, StructuralObservationCarrier)
        )
    return tuple(
        sorted(
            observations,
            key=lambda item: (item.file_path, item.line, item.owner_symbol),
        )
    )


def build_observation_graph(modules: list[ParsedModule]) -> ObservationGraph:
    """Build one observation graph from a list of parsed modules."""
    observations: list[StructuralObservation] = []
    for module in modules:
        observations.extend(collect_structural_observations(module))
    return ObservationGraph(tuple(observations))


def _is_public_observation_graph_export(name: str, value: object) -> bool:
    return bool(
        not name.startswith("_")
        and getattr(value, "__module__", None) == __name__
        and (isinstance(value, type) or callable(value))
    )


__all__ = sorted(
    name
    for name, value in globals().items()
    if _is_public_observation_graph_export(name, value)
)
