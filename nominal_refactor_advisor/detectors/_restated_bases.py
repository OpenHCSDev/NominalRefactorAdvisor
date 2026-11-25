"""A base another base of the same class already supplies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
)
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.needed_input import NeedsInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.restated_bases import (
    RestatedBaseFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@ScanCache.cached
def _base_restated_base_slot() -> list[
    tuple[
        tuple[CompactModuleClassProjection, ...],
        CompactClassFamilyIndex,
        dict[str, RestatedBaseCandidate],
    ]
    | None
]:
    return [None]


@dataclass(frozen=True)
class RestatedBaseCandidate:
    """One class and the bases it lists that another of its bases supplies."""

    indexed: CompactIndexedClass
    restated: tuple[tuple[str, str], ...]
    """Each restated base, with the base whose order already holds it."""

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (SourceLocation(self.indexed.file_path, self.indexed.line, self.indexed.qualname),)


class RestatedBaseDetector(
    CompactClassRepositoryCandidateDetector[RestatedBaseCandidate],
    RestatedBaseFindingRecipeSynthesizer,
):
    """A class listing a base its other bases already supply."""

    finding_spec = finding_spec_template(
        PatternId.RESTATED_BASE,
        "A class lists a base another of its bases supplies",
        "The class names a base that another of its bases already has in its order, and its order is the same without the name: the base list restates an ancestry the other base declares, a second copy that has to be kept in step with it.",
        "the base list without the restated base",
        "a plain-name base that another direct base's C3 order holds, whose removal leaves the class's C3 order unchanged",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[RestatedBaseCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )

        def _candidate_for(indexed: CompactIndexedClass) -> RestatedBaseCandidate | None:
            if len(indexed.base_references) < 2 or len(indexed.base_argument_sources) != len(
                indexed.base_references
            ):
                return None
            try:
                restated = verdicts.restated_bases(
                    indexed.symbol,
                    tuple(not arguments for arguments in indexed.base_argument_sources),
                )
                if not restated:
                    return None
                listed = verdicts.bases(indexed.symbol)
            except NeedsInput:
                return None
            return RestatedBaseCandidate(
                indexed,
                tuple((listed[index], supplier) for index, supplier in restated),
            )

        if ScanCache.active() is not None:
            slot = _base_restated_base_slot()
            base_entry = slot[0]
            if base_entry is not None:
                base_projs, base_index, base_by_symbol = base_entry
                base_by_path = {p.file_path: p for p in base_projs}
                cur_projs = tuple(context.projections)
                cur_by_path = {p.file_path: p for p in cur_projs}
                changed = tuple(
                    p for p in cur_projs if base_by_path.get(p.file_path) is not p
                )
                removed = tuple(
                    p for path, p in base_by_path.items() if path not in cur_by_path
                )
                if len(changed) + len(removed) <= 64:
                    old_by_sym = {
                        c.symbol: c
                        for p in (
                            *(
                                base_by_path[p.file_path]
                                for p in changed
                                if p.file_path in base_by_path
                            ),
                            *removed,
                        )
                        for c in p.classes
                    }
                    new_by_sym = {c.symbol: c for p in changed for c in p.classes}

                    def _rb_sig(c: CompactIndexedClass) -> tuple[object, ...]:
                        return (
                            c.base_references,
                            c.base_argument_sources,
                            c.base_references_are_complete,
                            c.resolved_base_symbols,
                            c.declared_base_names,
                            c.mro_bases_are_static,
                            c.star_resolution,
                        )

                    semantically_changed: set[str] = set()
                    line_shifted: dict[str, CompactIndexedClass] = {}
                    for sym in old_by_sym.keys() | new_by_sym.keys():
                        old_c = old_by_sym.get(sym)
                        new_c = new_by_sym.get(sym)
                        if old_c is None or new_c is None or _rb_sig(old_c) != _rb_sig(new_c):
                            semantically_changed.add(sym)
                        elif old_c.line != new_c.line:
                            line_shifted[sym] = new_c
                    affected: set[str] = set(semantically_changed)
                    for sym in semantically_changed:
                        affected.update(base_index.descendant_symbols(sym))
                        affected.update(context.class_index.descendant_symbols(sym))
                    classes = context.class_index.classes_by_symbol
                    by_symbol = {
                        sym: (
                            replace(cand, indexed=line_shifted[sym])
                            if sym in line_shifted
                            else cand
                        )
                        for sym, cand in base_by_symbol.items()
                        if sym in classes and sym not in affected
                    }
                    for sym in affected:
                        indexed = classes.get(sym)
                        if indexed is None:
                            continue
                        cand = _candidate_for(indexed)
                        if cand is not None:
                            by_symbol[sym] = cand
                    return tuple(
                        sorted(
                            by_symbol.values(),
                            key=lambda candidate: (
                                candidate.indexed.file_path,
                                candidate.indexed.line,
                            ),
                        )
                    )

        by_symbol_full: dict[str, RestatedBaseCandidate] = {}
        for indexed in context.class_index.classes_by_symbol.values():
            cand = _candidate_for(indexed)
            if cand is not None:
                by_symbol_full[indexed.symbol] = cand
        if ScanCache.active() is not None:
            slot = _base_restated_base_slot()
            if slot[0] is None:
                slot[0] = (
                    tuple(context.projections),
                    context.class_index,
                    by_symbol_full,
                )
        return tuple(
            sorted(
                by_symbol_full.values(),
                key=lambda candidate: (candidate.indexed.file_path, candidate.indexed.line),
            )
        )

    def _finding_for_candidate(self, candidate: RestatedBaseCandidate) -> RefactorFinding:
        return self.build_finding(
            f"`{candidate.indexed.qualname}` lists "
            + ", ".join(
                f"`{base.rpartition('.')[2]}`, which `{supplier.rpartition('.')[2]}` already supplies"
                for base, supplier in candidate.restated
            )
            + ": its order is the same without it.",
            candidate.evidence,
        )


__all__ = ("RestatedBaseCandidate", "RestatedBaseDetector")
