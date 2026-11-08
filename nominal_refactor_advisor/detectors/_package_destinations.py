"""Flat modules a destination table sends into a package."""

from __future__ import annotations

from collections.abc import Sequence

from ..class_index import CompactModuleClassProjection, CompactModuleClassProjectionFamily
from ..models import RefactorFinding
from ..patterns import PatternId
from ..refactors.package_move import PackageMoveFindingRecipeSynthesizer
from ..taxonomy import CapabilityTag, ObservationTag
from ..ast_tools import ParsedModuleSet
from ._base import (
    CompactModuleProjectionDetectorMixin,
    DetectorConfig,
    IssueDetector,
    high_confidence_certified_spec,
)


class FlatModulePackageDestinationDetector(
    PackageMoveFindingRecipeSynthesizer,
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
):
    """One finding per flat module the run's destination tables send somewhere."""

    module_projection_family = CompactModuleClassProjectionFamily
    finding_spec = high_confidence_certified_spec(
        PatternId.FLAT_MODULE_WITH_DECLARED_PACKAGE,
        "A flat module belongs in the package its destination table declares",
        "The destination table declares the package this module's behaviour "
        "belongs to, so the flat path restates an ownership the table already owns.",
        "the module's declarations in its package and the flat module dissolved",
        "a flat module of a root package that a destination-table row names by "
        "file or by a class it alone declares",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _collect_findings(
        self, modules: ParsedModuleSet, config: DetectorConfig
    ) -> list[RefactorFinding]:
        return self._findings_from_compact_projections(
            type(self).compact_module_projections(modules), config
        )

    def _findings_from_compact_projections(
        self,
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self.destination_findings(projections, config.package_destination_tables)
