"""Declarations a composition root's layering places above the code reading them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.composition_layering import (
    CompositionLayeringModuleFacts,
    CompositionLayeringModuleFamily,
    CompositionRootLayering,
    LayerRead,
)
from nominal_refactor_advisor.detectors._base import (
    CompactProjectionCandidateDetector,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.hidden_import_cuts import LayeringViolationFindingRecipeSynthesizer
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class LayeringViolationCandidate:
    """One misplaced declaration, where it is declared, each read of it from
    below, and the lowest layer holding every reader."""

    module: str
    name: str
    file_path: str
    lineno: int
    reads: tuple[LayerRead, ...]
    home: str | None


class LayeringViolationDetector(
    LayeringViolationFindingRecipeSynthesizer,
    CompactProjectionCandidateDetector[
        CompositionLayeringModuleFacts,
        LayeringViolationCandidate,
    ],
):
    """A declaration only a higher service loads, read by code a lower one loads."""

    module_projection_family = CompositionLayeringModuleFamily
    candidate_type = LayeringViolationCandidate
    finding_spec = finding_spec_template(
        PatternId.LAYERING_VIOLATION,
        "A declaration sits above the composition-root layer that reads it",
        "The composition root builds this service from the services it hands it, and code a service it is built on loads reads this declaration: through a hidden import, or because the declaration shares the higher service's module.",
        "the declaration in a module of the lowest layer that loads every module reading it",
        "a declared composition root's constructor arguments and each service's load closure",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.PARTIAL_VIEW,),
    )

    def _candidates_from_compact_projections(
        self,
        projections: Sequence[CompositionLayeringModuleFacts],
        config: DetectorConfig,
    ) -> tuple[LayeringViolationCandidate, ...]:
        del config
        layering = CompositionRootLayering.from_module_facts(projections)
        if not layering.reads:
            return ()
        paths = {projection.module_name: projection.file_path for projection in projections}
        named = {(read.location, read.declaration[0]) for read in layering.reads if read.declaration[1] is not None}
        candidates: list[LayeringViolationCandidate] = []
        for (module, name), reads in sorted(layering.by_declaration.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            if name is None and all((read.location, module) in named for read in reads):
                continue
            file_path = paths.get(module)
            if file_path is None:
                continue
            home = layering.home((module, name))
            candidates.append(
                LayeringViolationCandidate(
                    module,
                    name or module,
                    file_path,
                    1 if name is None or (line := layering.bindings.lineno((module, name))) is None else line,
                    reads,
                    None if home is None else home[1],
                )
            )
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: LayeringViolationCandidate) -> RefactorFinding:
        layers = sorted({f"`{read.lower}` (below `{read.higher}`)" for read in candidate.reads})
        return self.build_finding(
            f"`{candidate.module if candidate.name == candidate.module else f'{candidate.module}.{candidate.name}'}` is read from {', '.join(layers)} at "
            + ", ".join(f"{read.location} {read.symbol}".rstrip() for read in candidate.reads)
            + (
                f"; its home is the `{candidate.home}` layer."
                if candidate.home is not None
                else "; no layer below every reader loads every one of them."
            ),
            (
                SourceLocation(candidate.file_path, candidate.lineno, candidate.name),
                *(SourceLocation(read.file_path, read.lineno, read.symbol or read.reader) for read in candidate.reads),
            ),
        )


__all__ = ("LayeringViolationCandidate", "LayeringViolationDetector")
