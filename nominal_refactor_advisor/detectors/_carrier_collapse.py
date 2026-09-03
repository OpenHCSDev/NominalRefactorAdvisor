"""Proof-gated findings for collapsing flat fields into nominal carriers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from ..ast_tools import CollectedFamily
from ..carrier_collapse import ClosedCarrierCollapseComponent
from ..carrier_expansion import DeclaredCarrierExpansionBuilder
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..codemod import (
    CarrierCollapseFindingRecipeSynthesizer,
    CarrierCollapseOperationABC,
    CollapseClosedParameterConveyorOperation,
    CollapseDeclaredCarrierExpansionOperation,
    SourceRewriteTarget,
)
from ..models import ParameterThreadMetrics, RefactorFinding, SourceLocation
from ..parameter_conveyor import ClosedParameterConveyorComponentBuilder
from ..patterns import PatternId
from ..product_flow import (
    CompactProductFlowModuleProjection,
    CompactProductFlowModuleProjectionFamily,
)
from ..product_flow_authority import CompactProductFlowRepository
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CompactMultiProjectionCandidateDetector,
    DetectorConfig,
    high_confidence_certified_spec,
)


def _compact_product_flow_repository(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> CompactProductFlowRepository:
    del config
    return CompactProductFlowRepository(
        product_projections=cast(
            tuple[CompactProductFlowModuleProjection, ...],
            projections_by_family[CompactProductFlowModuleProjectionFamily],
        ),
        class_projections=cast(
            tuple[CompactModuleClassProjection, ...],
            projections_by_family[CompactModuleClassProjectionFamily],
        ),
    )


class CarrierCollapseCandidateDetector(
    CarrierCollapseFindingRecipeSynthesizer,
    CompactMultiProjectionCandidateDetector[ClosedCarrierCollapseComponent],
    ABC,
):
    """Share product-flow authority and finding assembly across collapse forms."""

    module_projection_families = (
        CompactProductFlowModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        _compact_product_flow_repository
    )

    def _candidates_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        return self._proven_components(
            _compact_product_flow_repository(projections_by_family, config)
        )

    def _candidates_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        del projections_by_family, config
        return self._proven_components(
            CompactProductFlowRepository.require(context)
        )

    @abstractmethod
    def _proven_components(
        self,
        repository: CompactProductFlowRepository,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        raise NotImplementedError

    def _finding_for_candidate(
        self,
        component: ClosedCarrierCollapseComponent,
    ) -> RefactorFinding:
        authority = component.authority
        authority_location = SourceLocation(
            authority.file_path,
            authority.line,
            authority.class_symbol,
        )
        participant_locations = tuple(
            SourceLocation(
                participant.context.file_path,
                participant.declaration.line,
                participant.symbol,
            )
            for participant in component.participants
        )
        call_locations = tuple(
            SourceLocation(
                edge.resolved_call.context.file_path,
                edge.resolved_call.call.line,
                edge.resolved_call.context.owner_symbol,
            )
            for edge in component.edges
        )
        return self.build_finding(
            (
                f"`{authority.class_symbol}` is flattened across "
                f"{len(component.participants)} private participant(s) and "
                f"{len(call_locations)} complete call edge(s)."
            ),
            (authority_location, *participant_locations, *call_locations),
            authority_evidence=authority_location,
            metrics=ParameterThreadMetrics(
                function_count=len(component.participants),
                shared_parameter_count=len(authority.field_names),
                shared_parameter_names=authority.field_names,
            ),
        )


class ClosedParameterConveyorDetector(CarrierCollapseCandidateDetector):
    """Expose constructor-derived call families proven safe to collapse."""

    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Closed parameter conveyor should use its existing nominal carrier",
        "A complete private call family repeatedly transports every field of one "
        "existing product authority while the product is already constructed at "
        "its roots. The proven component can carry that authority directly without "
        "preserving parallel flat parameters.",
        "one existing nominal carrier across the complete participating call family",
        "every product field is transported injectively through one closed call component",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.KEYWORD_MAPPING,
        ),
    )

    @classmethod
    def carrier_collapse_operation(
        cls,
        target: SourceRewriteTarget,
    ) -> CarrierCollapseOperationABC:
        return CollapseClosedParameterConveyorOperation(target=target)

    def _proven_components(
        self,
        repository: CompactProductFlowRepository,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        return ClosedParameterConveyorComponentBuilder(repository).proven_components()


class DeclaredCarrierExpansionDetector(CarrierCollapseCandidateDetector):
    """Expose declaration-typed field expansions proven safe to collapse."""

    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Declared carrier expansion should preserve its nominal carrier",
        "A complete private call family expands every field of one product authority "
        "from a value whose declared result type already proves that carrier. The "
        "proven component can preserve the carrier through every call instead of "
        "recreating a parallel flat parameter surface.",
        "one declared nominal carrier across the complete participating call family",
        "every carrier field is projected injectively through one closed call component",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.KEYWORD_MAPPING,
        ),
    )

    @classmethod
    def carrier_collapse_operation(
        cls,
        target: SourceRewriteTarget,
    ) -> CarrierCollapseOperationABC:
        return CollapseDeclaredCarrierExpansionOperation(target=target)

    def _proven_components(
        self,
        repository: CompactProductFlowRepository,
    ) -> tuple[ClosedCarrierCollapseComponent, ...]:
        return DeclaredCarrierExpansionBuilder(repository).proven_components()
