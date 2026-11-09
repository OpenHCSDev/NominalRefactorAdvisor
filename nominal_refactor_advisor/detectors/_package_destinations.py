"""Flat modules a destination table sends into a package."""

from __future__ import annotations

from collections.abc import Sequence

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.package_move import (
    DestinationTable,
    PackageMoveFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._base import (
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
            DestinationTable.from_texts(config.package_destination_tables).module_headers(
                tuple(modules), type(self).compact_module_projections
            ),
            config,
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
