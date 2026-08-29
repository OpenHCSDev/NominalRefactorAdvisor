"""Nominal registry-state refinements used by planner projections."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Iterable


class RegistryNormalFormStage(ABC):
    """One refinement in the registry normal-form path.

    Subclassing carries prerequisite order. A planner can therefore project
    any observed subset through the most-refined stage's MRO without a second
    ordered catalog.
    """

    normal_form: ClassVar[str]
    stage_label: ClassVar[str]
    step_template: ClassVar[str]

    @classmethod
    def plan_step(cls, subsystem: str) -> str:
        return cls.step_template.format(subsystem=subsystem)

    @classmethod
    def ordered_path(
        cls,
        stages: Iterable[type["RegistryNormalFormStage"]],
    ) -> tuple[type["RegistryNormalFormStage"], ...]:
        selected = frozenset(stages)
        if not selected:
            return ()
        most_refined = tuple(
            stage
            for stage in selected
            if all(stage is other or issubclass(stage, other) for other in selected)
        )
        if len(most_refined) != 1:
            stage_names = ", ".join(sorted(stage.__name__ for stage in selected))
            raise ValueError(
                "Registry normal-form stages must form one nominal refinement "
                f"path; got {stage_names}"
            )
        return tuple(
            stage
            for stage in reversed(most_refined[0].__mro__)
            if stage in selected
        )


class CanonicalRegistryIdentityStage(RegistryNormalFormStage):
    normal_form = "typed_record_table"
    stage_label = "repair injectivity"
    step_template = (
        "Repair `{subsystem}` registry injectivity first: give each concrete "
        "implementation one canonical key and move semantic aliases into an "
        "explicit alias projection."
    )


class ProvenRegistryMaturityStage(CanonicalRegistryIdentityStage):
    normal_form = "typed_record_table"
    stage_label = "demote premature registry"
    step_template = (
        "Demote unstable registry infrastructure in `{subsystem}` to a typed "
        "table or local strategy map until key cases, lookup lifecycle, and "
        "consumer fanout are all proven."
    )


class MetaclassPromotionEligibleStage(ABC):
    """Marker for registry states that can proceed toward metaclass ownership."""


class SingleRegistryAuthorityStage(
    ProvenRegistryMaturityStage,
    MetaclassPromotionEligibleStage,
):
    normal_form = "generated_projection_surface"
    stage_label = "choose authority and derive projection"
    step_template = (
        "Choose one injective registry authority in `{subsystem}` and derive "
        "the parallel keyed table as a generated projection, or demote the "
        "family if behavior is only metadata."
    )


class DerivedRegistryProjectionStage(SingleRegistryAuthorityStage):
    normal_form = "generated_projection_surface"
    stage_label = "merge keyed projections"
    step_template = (
        "Merge parallel keyed tables in `{subsystem}` into one finite axis "
        "catalog and derive each table surface from that catalog."
    )


class UnifiedRegistryAxisFamilyStage(DerivedRegistryProjectionStage):
    normal_form = "auto_registered_abc"
    stage_label = "merge keyed families"
    step_template = (
        "Merge sibling keyed registry families in `{subsystem}` into one "
        "shared ABC/mixin lattice over the common key axis."
    )


class MetaclassRegisteredRegistryStage(UnifiedRegistryAxisFamilyStage):
    normal_form = "auto_registered_abc"
    stage_label = "promote mature injective registry"
    step_template = (
        "Promote the mature injective registry in `{subsystem}` to "
        "`AutoRegisterMeta`; implementation classes should retain only "
        "canonical key attributes and behavior hooks."
    )


@dataclass(frozen=True)
class RegistryNormalFormPath:
    """One MRO-derived projection of observed registry refinement stages."""

    stages: tuple[type[RegistryNormalFormStage], ...]

    @classmethod
    def from_stages(
        cls,
        stages: Iterable[type[RegistryNormalFormStage]],
    ) -> "RegistryNormalFormPath":
        return cls(RegistryNormalFormStage.ordered_path(stages))

    @property
    def canonical_clause(self) -> str:
        if not self.stages:
            return ""
        stage_labels = " -> ".join(stage.stage_label for stage in self.stages)
        return (
            f"registry normal-form path ({stage_labels}) ending in "
            f"`{self.stages[-1].normal_form}`"
        )

    def plan_steps(self, subsystem: str) -> tuple[str, ...]:
        steps = tuple(stage.plan_step(subsystem) for stage in self.stages)
        if any(
            not issubclass(stage, MetaclassPromotionEligibleStage)
            for stage in self.stages
        ):
            return steps + (
                f"After the blocking registry stages are fixed in `{subsystem}`, rerun NRA before promoting any registry to metaclass registration.",
            )
        return steps
