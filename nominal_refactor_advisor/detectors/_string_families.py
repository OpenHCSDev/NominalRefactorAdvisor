"""A nominal family kept by hand: one string key selecting behaviour at several sites."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.class_identity_observations import (
    DeclaredIdentities,
    DeclaredIdentityFamily,
    MemberSpellings,
)
from nominal_refactor_advisor.detectors._base import (
    CompactMultiModuleProjectionDetectorMixin,
    CompactProjectionGroups,
    DetectorConfig,
    IssueDetector,
    finding_spec_template,
)
from nominal_refactor_advisor.models import FindingMetrics, ImpactDelta, RefactorFinding
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.string_families import (
    StringFamily,
    StringKeySiteFamily,
)
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag

LISTED = 6
"""Sites and members a finding's text names; its evidence and metrics keep them all."""


@dataclass(frozen=True)
class StringFamilyMetrics(FindingMetrics):
    """The family a key spells: its members (literal groups), the sites that are its
    methods, and whether the member count is the exact minimum or a greedy bound."""

    key: str
    members: tuple[tuple[str, ...], ...]
    """One tuple per member: the literals that spell it (more than one: a merge)."""
    methods: tuple[str, ...]
    """One per selecting site: the function whose branches become each member's override."""
    minimum: int
    """The fewest members that keep every site's answer (paper1 Thm 3.2)."""
    lower_bound: int
    exact: bool
    confidence: float

    @property
    def impact_delta(self) -> ImpactDelta:
        arms = len(self.methods) * sum(len(member) for member in self.members)
        return ImpactDelta(
            lower_bound_removable_loc=arms,
            upper_bound_removable_loc=2 * arms,
            loci_of_change_before=arms,
            loci_of_change_after=1,
        )


class StringKeyedFamilyDetector(
    CompactMultiModuleProjectionDetectorMixin,
    IssueDetector,
):
    """One key compared with overlapping string literals in two or more functions.

    A string stands in for a nominal identity in one of two ways, and each has one
    owner. When the scan declares a family whose `key` attribute spells every literal,
    the strings are second copies of declared members: `RestatedIdentityDetector`
    owns them and the repair references the members. Otherwise the family is kept
    only by hand, and this detector recovers it; literals a declared family already
    spells name the family to extend (paper1 Prop 3.1, Thm 3.2).
    """

    module_projection_families = (StringKeySiteFamily, DeclaredIdentityFamily)
    compact_shared_group_context_builder = staticmethod(
        DeclaredIdentities.from_projection_groups
    )
    finding_spec = finding_spec_template(
        PatternId.STRING_KEYED_FAMILY,
        "A string key restates a family at several sites",
        "Each site that compares the key with string literals restates which strings belong to the family and what each one does; a member added later must be found at every site, and two sites can disagree. The strings no site tells apart are one member spelled several ways; the fewest members that keep every site's answer is the chromatic number of the graph joining literals some site sends to different branches (paper1 Thm 3.2).",
        "a class family with one member per literal group and one method per selecting site, the strings kept only in the boundary decoder",
        "a key compared with string literals in two or more functions, two or more literals in all, the sites' literals overlapping",
        (CapabilityTag.CLOSED_FAMILY_DISPATCH, CapabilityTag.NOMINAL_IDENTITY),
        (ObservationTag.STRING_DISPATCH,),
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: DeclaredIdentities,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        families = StringFamily.joined(
            site
            for module in StringKeySiteFamily.modules_in(projections_by_family)
            for site in module.sites
        )
        found = []
        for family in sorted(families, key=lambda family: -family.weight):
            declared = context.members.get(family.key)
            spelled = (
                set()
                if declared is None
                else {value for value, _location in declared.members}
            )
            if self.admits(family) and not set(family.literals) <= spelled:
                partly = declared if spelled & set(family.literals) else None
                found.append(self._finding(family, partly, spelled))
        return found

    @staticmethod
    def admits(family: StringFamily) -> bool:
        """Two literals or more across two functions or more, and not every site a
        decoder: one function is StringKindDispatchDetector's, one literal is a guard,
        and a decoder is where strings belong. A family split one test per function is
        still one family (paper1 Thm 3.2 counts members, not where their arms sit)."""

        return (
            len(family.literals) >= 2
            and len({site.symbol for site in family.sites}) >= 2
            and family.boundary_share < 1.0
        )

    def _finding(
        self,
        family: StringFamily,
        extended: MemberSpellings | None,
        spelled: set[str],
    ) -> RefactorFinding:
        members = family.members
        listed = ", ".join(
            "{" + ", ".join(repr(literal) for literal in member) + "}" if len(member) > 1 else repr(member[0])
            for member in members[:LISTED]
        )
        methods = tuple(dict.fromkeys(site.symbol.rpartition(".")[2] for site in family.sites))
        count = (
            f"{len(members)} member(s), the minimum"
            if family.exact
            else f"{len(members)} member(s) by greedy colouring, at least {family.clique}"
        )
        merged = (
            f"; {len(family.merges)} member(s) are spelled more than one way"
            if family.merges
            else ""
        )
        boundary = (
            f"; {sum(site.boundary for site in family.sites)} of the sites decode a foreign value and keep their strings"
            if family.boundary_share
            else ""
        )
        if extended is None:
            owner = "Declare the family base with one method per site, one member per literal group overriding each"
            declared_at = ()
        else:
            missing = ", ".join(repr(literal) for literal in family.literals if literal not in spelled)
            bases = ", ".join(f"`{base}`" for base in sorted(extended.family_bases)[:LISTED])
            owner = (
                f"The family is declared below {bases}, whose `{family.key}` spells some of the literals; "
                f"add a member for each of {missing}, declare one method per site on the base, override it in each member"
            )
            declared_at = extended.declared_at
        return self.build_finding(
            f"`.{family.key}` selects behaviour by {len(family.literals)} string(s) at "
            f"{len(family.sites)} site(s) in {len(methods)} function(s) "
            f"({', '.join(f'`{name}`' for name in methods[:LISTED])}): the family needs {count}: "
            f"{listed}{merged}{boundary}. Confidence {family.confidence:.2f} (uncalibrated). "
            f"{owner}, and decode the strings once at the boundary (`member_named`). "
            "No codemod operation moves a multi-site family yet; this finding is the prescription.",
            (*(site.location for site in family.sites), *declared_at),
            metrics=StringFamilyMetrics(
                key=family.key,
                members=members,
                methods=methods,
                minimum=len(members),
                lower_bound=family.clique,
                exact=family.exact,
                confidence=round(family.confidence, 3),
            ),
        )


__all__ = ("StringFamilyMetrics", "StringKeyedFamilyDetector")
