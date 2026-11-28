"""Detect hand-typed dotted class-name string rosters restating capability membership (MEMB-2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.class_index import CompactClassFamilyIndex
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCandidateDetector,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.roster_to_capability_mixin import (
    extract_dotted_class_rosters,
    minimal_sound_ancestor_cover,
    module_compares_roster_against_dotted_names,
    most_specific_common_ancestor,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class DottedClassRosterCandidate:
    """One module- or class-level literal listing dotted class paths in the repository."""

    file_path: str
    line: int
    name: str
    qualname: str
    total_entries: int
    resolved_symbols: tuple[str, ...]
    unresolved_entries: tuple[str, ...]
    common_family_symbol: str | None
    minimal_cover_symbols: tuple[str, ...]
    compared_with_dotted_name: bool


class DottedClassRosterDetector(
    CrossModuleCandidateDetector[DottedClassRosterCandidate]
):
    """Detect `frozenset`/`set`/`tuple`/`list` literals of dotted class-path strings (MEMB-2)."""

    finding_spec = finding_spec_template(
        PatternId.DOTTED_CLASS_ROSTER,
        "Dotted class-name string roster restates capability membership (MEMB-2)",
        "A hand-typed collection of `'pkg.mod.ClassName'` strings compared against `f'{cls.__module__}.{cls.__name__}'` duplicates class membership outside the declarations and drifts whenever subclasses are added or renamed.",
        "a capability mixin inherited by the minimal ancestor cover and queried via `issubclass` / `Family.members_with(Mixin)` (`roster_to_capability_mixin`)",
        "literal collection of `>= 2` dotted class-path strings resolving to classes in the repository class index",
        (CapabilityTag.CAPABILITY_MARKER_IDENTITY, CapabilityTag.CLASS_LEVEL_REGISTRATION),
        (ObservationTag.CAPABILITY_MARKER, ObservationTag.CLASS_FAMILY),
    )

    def _candidate_items(
        self,
        modules: ParsedModuleSet,
        config: DetectorConfig,
    ) -> Sequence[DottedClassRosterCandidate]:
        del config
        index = CompactClassFamilyIndex.from_modules(modules)
        candidates: list[DottedClassRosterCandidate] = []
        for module in modules:
            for roster in extract_dotted_class_rosters(module):
                resolved = tuple(
                    dict.fromkeys(
                        entry
                        for entry in roster.entries
                        if entry in index.classes_by_symbol
                    )
                )
                unresolved = tuple(
                    entry
                    for entry in roster.entries
                    if entry not in index.classes_by_symbol
                )
                if len(resolved) < 2 or len(resolved) * 2 < len(roster.entries):
                    continue
                common_family = most_specific_common_ancestor(index, resolved)
                compared = module_compares_roster_against_dotted_names(
                    module.module, roster.name
                )
                if common_family is None and not compared:
                    continue
                cover = minimal_sound_ancestor_cover(
                    index,
                    frozenset(resolved),
                    family_root=common_family,
                )
                candidates.append(
                    DottedClassRosterCandidate(
                        file_path=roster.file_path,
                        line=roster.line,
                        name=roster.name,
                        qualname=roster.qualname,
                        total_entries=len(roster.entries),
                        resolved_symbols=resolved,
                        unresolved_entries=unresolved,
                        common_family_symbol=common_family,
                        minimal_cover_symbols=cover,
                        compared_with_dotted_name=compared,
                    )
                )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.file_path, item.line, item.qualname),
            )
        )

    def _finding_for_candidate(
        self, candidate: DottedClassRosterCandidate
    ) -> RefactorFinding:
        family_note = (
            f" under `{candidate.common_family_symbol}`"
            if candidate.common_family_symbol is not None
            else ""
        )
        stale_note = (
            f" ({len(candidate.unresolved_entries)} stale/unresolved)"
            if candidate.unresolved_entries
            else ""
        )
        return self.build_finding(
            f"`{candidate.qualname}` hand-lists {len(candidate.resolved_symbols)}/"
            f"{candidate.total_entries} dotted class paths{family_note}{stale_note} "
            f"(MEMB-2); a capability mixin covers all {len(candidate.resolved_symbols)} "
            f"classes with {len(candidate.minimal_cover_symbols)} base declaration(s) "
            "via `roster_to_capability_mixin`.",
            (SourceLocation(candidate.file_path, candidate.line, candidate.qualname),),
        )


__all__ = (
    "DottedClassRosterCandidate",
    "DottedClassRosterDetector",
)
