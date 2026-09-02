"""Proof-gated parameter-conveyor findings."""

from __future__ import annotations

from typing import cast

from ..ast_tools import CollectedFamily
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..models import ParameterThreadMetrics, RefactorFinding, SourceLocation
from ..parameter_conveyor import (
    ClosedParameterConveyorComponent,
    ClosedParameterConveyorComponentBuilder,
)
from ..patterns import PatternId
from ..product_flow import (
    CompactProductFlowModuleProjection,
    CompactProductFlowModuleProjectionFamily,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    CompactMultiProjectionCandidateDetector,
    DetectorConfig,
    high_confidence_certified_spec,
)


class ClosedParameterConveyorDetector(
    CompactMultiProjectionCandidateDetector[ClosedParameterConveyorComponent],
):
    """Expose only whole call families proven safe to collapse to one carrier."""

    module_projection_families = (
        CompactProductFlowModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Closed parameter conveyor should use its existing nominal carrier",
        "A complete private call family repeatedly transports every field of an "
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

    def _candidates_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> tuple[ClosedParameterConveyorComponent, ...]:
        del config
        return ClosedParameterConveyorComponentBuilder.from_projections(
            cast(
                tuple[CompactProductFlowModuleProjection, ...],
                projections_by_family[CompactProductFlowModuleProjectionFamily],
            ),
            cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
        ).proven_components()

    def _finding_for_candidate(
        self,
        component: ClosedParameterConveyorComponent,
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
            for edge in (*component.root_edges, *component.forwarding_edges)
        )
        call_count = len(call_locations)
        return self.build_finding(
            (
                f"`{authority.class_symbol}` is reconstructed across "
                f"{len(component.participants)} private participant(s) and "
                f"{call_count} complete call edge(s)."
            ),
            (authority_location, *participant_locations, *call_locations),
            authority_evidence=authority_location,
            metrics=ParameterThreadMetrics(
                function_count=len(component.participants),
                shared_parameter_count=len(authority.field_names),
                shared_parameter_names=authority.field_names,
            ),
        )
