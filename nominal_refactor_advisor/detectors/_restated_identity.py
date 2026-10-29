"""Restated identity: a string that spells a class, module, family member or enum value the scan declares."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


from ..class_identity_observations import (
    DeclaredIdentities,
    DeclaredIdentityFamily,
    IdentitySpelling,
    IdentitySpellingFamily,
    ModuleDeclarations,
)
from ..codemod_declaration_reference import (
    DeclarationReferenceFindingRecipeSynthesizer,
    ReferenceClassNameOperation,
    ReferenceEnumMemberOperation,
    ReferenceImportedModuleOperation,
    UnquoteResolvedTypeNamesOperation,
)
from ..models import DeclarationReferenceMetrics, RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..restated_fact_observations import RestatingSite
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import CompactProjectionGroups
from ._restated_facts import JoinedRestatedFactDetector, RestatedFact, _listed


def _declared_identities(
    projections_by_family: CompactProjectionGroups,
) -> DeclaredIdentities:
    return DeclaredIdentities.from_modules(
        DeclaredIdentityFamily.modules_in(projections_by_family)
    )


class RestatedIdentityDetector(
    JoinedRestatedFactDetector[DeclaredIdentities, RestatingSite],
    DeclarationReferenceFindingRecipeSynthesizer,
):
    """A string literal that spells a declared class, module, family member or enum value in a position that uses it."""

    module_projection_families = (IdentitySpellingFamily, DeclaredIdentityFamily)
    restated = RestatedFact(
        fact="The string spells a declared identity (a class name, a module's dotted name, a family member's distinct spelling, or an enum member's value), a second copy of it that a rename or a same-spelled sibling breaks silently.",
        owner="The owner is the declaration itself: the class object, the imported module, a method the family declares, or the enum member.",
        factoring="Reference the declaration instead of spelling it: name the class (an unquoted annotation, or the class object passed or compared), import the module and pass it or its `__name__`, and replace a comparison of member spellings with the member class or a method the family declares, and write an enum member's value as `Enum.MEMBER.value`; keep the spelling only in the family's boundary decoder.",
        operations=(
            UnquoteResolvedTypeNamesOperation,
            ReferenceImportedModuleOperation,
            ReferenceClassNameOperation,
            ReferenceEnumMemberOperation,
        ),
    )
    finding_spec = restated.spec(
        PatternId.RESTATED_IDENTITY,
        "A string restates a declared identity",
        "the declared class, module or family member referenced directly instead of spelled",
        "a string literal equal to a class name, a scanned module's dotted name, a family member's distinct class keyword or constant, or a dunder or multi-word enum member value, in a type, comparison, match, argument, return, key, module constant or importer position",
        (CapabilityTag.NOMINAL_IDENTITY,),
        (ObservationTag.SEMANTIC_STRING_LITERAL,),
    )

    compact_shared_group_context_builder = staticmethod(_declared_identities)

    def findings_from_sites(
        self, sites: Iterable[RestatingSite], context: DeclaredIdentities
    ) -> list[RefactorFinding]:
        resolved = [
            found for site in sites if (found := context.resolve(site)) is not None
        ]
        return super().findings_from_sites(resolved, context.with_carriers(resolved))

    def describe(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> str:
        spelled = [
            f"{IdentitySpelling.positions()[site.detail[0]].spelling(site.detail)} "
            f"({site.detail[0]}, line {site.lineno})"
            for site in sites
        ]
        return (
            f"{sites[0].symbol} spells {sites[0].group} {len(sites)} time(s): "
            f"{_listed(spelled)}."
        )

    def owner(self, sites: Sequence[RestatingSite], context: DeclaredIdentities) -> str:
        declared = context.declarations_of(sites)
        where = _listed(
            f"`{location.symbol}` ({location.file_path}:{location.line})"
            for location in declared
        )
        return f"{self.restated.owner} Declared at {where}."

    def evidence(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> tuple[SourceLocation, ...]:
        return (*super().evidence(sites, context), *context.declarations_of(sites))

    def finding_context(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> dict[str, object]:
        carried = context.carried_rewrites(sites)
        return {"metrics": DeclarationReferenceMetrics(carried)} if carried else {}


__all__ = ("RestatedIdentityDetector",)
