"""Hubs whose declarations sit above the hidden imports that read them."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCandidateDetector,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.hidden_imports import ClosesCycle, HiddenImportPlan
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.hidden_import_cuts import (
    HiddenImportCutFindingRecipeSynthesizer,
    HubEdge,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class HiddenImportCycleCandidate:
    """One hub and the modules whose hidden imports of it close a cycle."""

    hub: str
    file_path: str
    readers: tuple[str, ...]


class HiddenImportCycleDetector(
    HiddenImportCutFindingRecipeSynthesizer,
    CrossModuleCandidateDetector[HiddenImportCycleCandidate],
):
    """A module other modules can import only by hiding it, because it loads them."""

    candidate_type = HiddenImportCycleCandidate
    finding_spec = finding_spec_template(
        PatternId.HIDDEN_IMPORT_CYCLE,
        "A hub's declarations sit above the modules that read them",
        "Modules import names from this hub under `TYPE_CHECKING` or inside functions, and importing them at the top would close a cycle, since the hub already loads them: the names are declared above their readers.",
        "the declarations the readers take, in a module below the hub that loads no reader",
        "a module that cycle-refused hidden imports read names from",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _candidate_items(
        self,
        modules: ParsedModuleSet,
        config: DetectorConfig,
    ) -> tuple[HiddenImportCycleCandidate, ...]:
        del config
        paths = {module.module_name: module.file_path for module in modules}
        readers: dict[str, set[str]] = defaultdict(set)
        for refusal in modules.fact(HiddenImportPlan).refused:
            if isinstance(refusal, ClosesCycle):
                readers[HubEdge.hub_of(refusal) or refusal.reaching].add(refusal.hidden.module.module_name)
        return tuple(
            HiddenImportCycleCandidate(hub, paths[hub], tuple(sorted(names)))
            for hub, names in sorted(readers.items())
            if hub in paths
        )

    def _finding_for_candidate(self, candidate: HiddenImportCycleCandidate) -> RefactorFinding:
        return self.build_finding(
            f"`{candidate.hub}` loads {len(candidate.readers)} module(s) that import it only by hiding the import: "
            + ", ".join(f"`{reader}`" for reader in candidate.readers)
            + ".",
            (SourceLocation(candidate.file_path, 1, candidate.hub),),
        )


__all__ = ("HiddenImportCycleCandidate", "HiddenImportCycleDetector")
