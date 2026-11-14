"""A class that registers its family by hand while a dependency already does (#71)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.installed_owners import (
    InstalledOwner,
    RegistrationOwners,
    RegistrationRole,
)
from nominal_refactor_advisor.owner_candidates import OwnerCandidate, ranked
from nominal_refactor_advisor.models import (
    InstalledOwnerRegistrationMetrics,
    RefactorFinding,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    finding_spec_template,
)


@dataclass(frozen=True)
class HandRolledRegistration:
    """A scanned class whose own creation path stores what it creates, and the
    installed classes that already do."""

    indexed_class: CompactIndexedClass
    role: RegistrationRole
    owners: tuple[InstalledOwner, ...]
    """The installed owners, ranked."""
    scope: RegistrationOwners


class HandRolledRegistrationDetector(
    CompactClassRepositoryCandidateDetector[HandRolledRegistration]
):
    """A class that stores each class of its family into a mapping itself, in a
    project whose installed dependencies already declare a class that does."""

    finding_spec = finding_spec_template(
        PatternId.AUTO_REGISTER_META,
        "A class registers its family by hand while a dependency already owns registration",
        "The class stores each class it creates (as a metaclass) or each subclass (from `__init_subclass__`) into a mapping itself. A package the project already depends on declares a class whose creation path does the same, read from its installed source. Keeping both keeps two registration algorithms whose uniqueness and inheritance rules drift apart.",
        "the installed owner, composed by the family root; the local registration deleted",
        "a scanned class whose creation path stores the created class into a mapping, beside an installed class whose creation path does",
        (CapabilityTag.CLASS_LEVEL_REGISTRATION, CapabilityTag.NOMINAL_IDENTITY),
        (ObservationTag.MANUAL_REGISTRATION, ObservationTag.CLASS_FAMILY),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[HandRolledRegistration, ...]:
        del config
        candidates: list[HandRolledRegistration] = []
        for indexed_class in sorted(
            context.class_index.classes_by_symbol.values(),
            key=lambda item: item.symbol,
        ):
            facts = indexed_class.registry_facts
            roles = RegistrationRole.played(
                facts.stores_created_class, facts.stores_subclass
            )
            if not roles:
                continue
            scope = context.cached(
                (RegistrationOwners, indexed_class.module_name, indexed_class.file_path),
                lambda: RegistrationOwners.for_package(
                    context.projections,
                    indexed_class.module_name,
                    Path(indexed_class.file_path),
                ),
            )
            owners = ranked(
                scope.candidates(frozenset({indexed_class.symbol}), roles, 0)
            )
            if owners:
                candidates.append(
                    HandRolledRegistration(indexed_class, roles[0], owners, scope)
                )
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: HandRolledRegistration) -> RefactorFinding:
        owner = candidate.owners[0]
        how = candidate.role.finding_text
        return self.build_finding(
            (
                f"`{candidate.indexed_class.qualname}` {how}. Installed owners "
                "(confidence uncalibrated): "
                + OwnerCandidate.ranked_text(candidate.owners)
                + ". Move: make the family inherit or compose "
                + f"`{owner.installed.symbol}` and "
                "delete this class's own registration. Considered: "
                + (", ".join(candidate.scope.distributions) or "no distributions")
                + f"; {candidate.scope.premise}"
                + "".join(f"; gap: {gap}" for gap in candidate.scope.gaps)
                + "."
            ),
            (candidate.indexed_class.location,),
            metrics=InstalledOwnerRegistrationMetrics(
                registration_site_count=1,
                class_names=(candidate.indexed_class.qualname,),
            ),
        )


__all__ = ("HandRolledRegistration", "HandRolledRegistrationDetector")
