"""Flat modules a destination table sends into a package."""

from __future__ import annotations

from collections.abc import Sequence

from ..ast_tools import ParsedModuleSet
from ..class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..models import RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..refactors.package_move import (
    DestinationTable,
    PackageMoveFindingRecipeSynthesizer,
)
from ..taxonomy import CapabilityTag, ObservationTag
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
        """The table reads only each module's name, path and classes, so the
        header-core view answers it; nothing settles field declarations."""

        return self._findings_from_compact_projections(
            CompactModuleClassProjectionFamily.header_core_projections(modules), config
        )

    def _findings_from_compact_projections(
        self,
        projections: Sequence[CompactModuleClassProjection],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        """Findings are this detector's answers, so they are computed here, where the
        cache key reads them, never on the answer-free synthesizer it inherits."""
        return [
            self.build_finding(
                f"`{destination.source_path}` belongs in "
                f"`{destination.destination_path}`, the package its destination "
                "table declares.",
                (SourceLocation(destination.source_path, 1, "<module>"),),
                metrics=destination.metrics(),
            )
            for destination in DestinationTable.from_texts(
                config.package_destination_tables
            ).destinations(projections)
        ]
