"""A member its class declares exactly as the class it would inherit it from."""

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
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.restated_overrides import (
    RestatedOverrideFindingRecipeSynthesizer,
    inherited_restatements,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@ScanCache.cached
def _base_restated_override_slot() -> list[
    tuple[
        tuple[CompactModuleClassProjection, ...],
        CompactClassFamilyIndex,
        dict[str, RestatedOverrideCandidate],
    ]
    | None
]:
    return [None]


@dataclass(frozen=True)
class RestatedOverrideCandidate:
    """One class and the members it restates, each with the class it would inherit it from."""

    indexed: CompactIndexedClass
    restated: tuple[tuple[str, str], ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (SourceLocation(self.indexed.file_path, self.indexed.line, self.indexed.qualname),)


class RestatedOverrideDetector(
    CompactClassRepositoryCandidateDetector[RestatedOverrideCandidate],
    RestatedOverrideFindingRecipeSynthesizer,
):
    """A class restating a member it would inherit unchanged."""

    finding_spec = finding_spec_template(
        PatternId.RESTATED_OVERRIDE,
        "A class restates a member it would inherit",
        "The class declares a member exactly as the next class in its order that declares it: the same declaration, bound to the same names, reading nothing of the class it sits in. Without the copy every lookup reaches the inherited one, so the copy is a second statement of one answer that has to be kept in step with it.",
        "the class without the restated member",
        "a member whose class-body normal form equals the next declaring class's, with neither reading its class",
        (CapabilityTag.MRO_ORDERING,),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[RestatedOverrideCandidate, ...]:
        del config
        verdicts = context.cached(
            ClassOverlapVerdicts,
            lambda: ClassOverlapVerdicts(tuple(context.projections), context.class_index),
        )
        if ScanCache.active() is not None:
            slot = _base_restated_override_slot()
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

                    def _ro_sig(c: CompactIndexedClass) -> tuple[object, ...]:
                        return (
                            c.base_references,
                            c.base_references_are_complete,
                            c.resolved_base_symbols,
                            c.declared_base_names,
                            c.mro_bases_are_static,
                            c.star_resolution,
                            c.abstract_method_names,
                            tuple(
                                (
                                    d.kind,
                                    d.name,
                                    d.digest,
                                    d.bindings,
                                    d.deferred,
                                    d.unbound,
                                    d.field_declaration,
                                    d.reads_its_class,
                                )
                                for d in c.body_declarations
                            ),
                        )

                    semantically_changed: set[str] = set()
                    line_shifted: dict[str, CompactIndexedClass] = {}
                    for sym in old_by_sym.keys() | new_by_sym.keys():
                        old_c = old_by_sym.get(sym)
                        new_c = new_by_sym.get(sym)
                        if old_c is None or new_c is None or _ro_sig(old_c) != _ro_sig(new_c):
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
                        if restated := inherited_restatements(verdicts, sym):
                            by_symbol[sym] = RestatedOverrideCandidate(indexed, restated)
                    return tuple(
                        sorted(
                            by_symbol.values(),
                            key=lambda candidate: (
                                candidate.indexed.file_path,
                                candidate.indexed.line,
                            ),
                        )
                    )

        by_symbol_full = {
            indexed.symbol: RestatedOverrideCandidate(indexed, restated)
            for indexed in context.class_index.classes_by_symbol.values()
            if (restated := inherited_restatements(verdicts, indexed.symbol))
        }
        if ScanCache.active() is not None:
            slot = _base_restated_override_slot()
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

    def _finding_for_candidate(self, candidate: RestatedOverrideCandidate) -> RefactorFinding:
        return self.build_finding(
            f"`{candidate.indexed.qualname}` declares "
            + ", ".join(
                f"`{name}` exactly as `{inherited.rpartition('.')[2]}` does"
                for name, inherited in candidate.restated
            )
            + ": lookup reaches the same declaration without it.",
            candidate.evidence,
        )


__all__ = ("RestatedOverrideCandidate", "RestatedOverrideDetector")
