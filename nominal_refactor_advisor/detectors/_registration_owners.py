"""A class that registers its family by hand while a dependency already does (#71)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..class_index import CompactIndexedClass
from ..installed_owners import (
    InstalledRegistrationOwner,
    RegistrationOwners,
    RegistrationRole,
)
from ..models import RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
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
    owners: tuple[InstalledRegistrationOwner, ...]
    scope: RegistrationOwners

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(
            self.indexed_class.file_path, self.indexed_class.line, self.indexed_class.symbol
        )


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
            role = (
                RegistrationRole.METACLASS
                if facts.stores_created_class
                else RegistrationRole.SUBCLASS_HOOK
                if facts.stores_subclass
                else None
            )
            if role is None:
                continue
            scope = context.cached(
                (RegistrationOwners, indexed_class.module_name, indexed_class.file_path),
                lambda: RegistrationOwners.for_package(
                    context.projections,
                    indexed_class.module_name,
                    Path(indexed_class.file_path),
                ),
            )
            owners = scope.owners_besides(indexed_class.symbol)
            if owners:
                candidates.append(
                    HandRolledRegistration(indexed_class, role, owners, scope)
                )
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: HandRolledRegistration) -> RefactorFinding:
        owner = candidate.owners[0]
        how = (
            "stores each class it creates in a mapping"
            if candidate.role is RegistrationRole.METACLASS
            else "stores each subclass in a mapping from `__init_subclass__`"
        )
        return self.build_finding(
            (
                f"`{candidate.indexed_class.qualname}` {how}. Installed "
                + "; ".join(owner.receipt for owner in candidate.owners)
                + f". Move: make the family inherit or compose `{owner.symbol}` and "
                "delete this class's own registration. Considered: "
                + (", ".join(candidate.scope.distributions) or "no distributions")
                + f"; {candidate.scope.premise}"
                + "".join(f"; gap: {gap}" for gap in candidate.scope.gaps)
                + "."
            ),
            (candidate.location,),
        )


__all__ = ("HandRolledRegistration", "HandRolledRegistrationDetector")
