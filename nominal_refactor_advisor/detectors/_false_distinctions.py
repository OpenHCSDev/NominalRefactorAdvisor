"""False distinctions: tests, classes and members that distinguish in name only.

Each detector here names code that tells cases apart and then treats them the
same, the fact that distinction restates, and the factoring that removes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..class_index import CompactModuleClassProjectionFamily
from ..codemod_false_distinctions import (
    CollapseIdenticalArmsOperation,
    EnumFoldFindingRecipeSynthesizer,
    FoldEnumMembersOperation,
    EmptySubclassFindingRecipeSynthesizer,
    RedirectEmptySubclassOperation,
    IdenticalArmsFindingRecipeSynthesizer,
    MergeTypeTestArmsOperation,
    UndistinguishedTypeTestFindingRecipeSynthesizer,
)
from ..false_distinction_observations import (
    EmptySubclassIndex,
    EnumMemberIndex,
    EnumUseSiteFamily,
    EmptySubclassSiteFamily,
    IdenticalArmsSiteFamily,
    UndistinguishedTypeArmsSiteFamily,
)
from ..patterns import PatternId
from ..restated_fact_observations import RestatingSite
from ..taxonomy import ObservationTag
from ._restated_facts import (
    JoinedRestatedFactDetector,
    RestatedFact,
    SingleFamilyRestatedFactDetector,
    _listed,
)


class IdenticalArmsDetector(
    SingleFamilyRestatedFactDetector, IdenticalArmsFindingRecipeSynthesizer
):
    """A branch whose every arm is the same code."""

    module_projection_family = IdenticalArmsSiteFamily
    restated = RestatedFact(
        fact="Every arm of the branch runs the same code, so its tests distinguish cases that the code then treats identically; the distinction exists in the tests alone.",
        owner="The owner is the one arm: it runs whatever the tests say.",
        factoring="Replace the branch with its one arm and delete the tests; a test whose evaluation can have an effect stays, so that branch is not collapsed.",
        operations=(CollapseIdenticalArmsOperation,),
    )
    finding_spec = restated.spec(
        PatternId.IDENTICAL_ARMS,
        "Every arm of a branch runs the same code",
        "the branch replaced by its one arm, its tests deleted",
        "an if/elif ladder with a final else, a conditional expression or a match ending in `case _:` whose arms are the same code, or `if test: S` followed directly by the same S that leaves the block",
        (),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        branches = _listed(
            f"line {site.lineno} on {' / '.join(f'`{test}`' for test in site.detail)}"
            for site in sites
        )
        return f"`{sites[0].symbol}` branches with identical arms: {branches}."


class UndistinguishedTypeTestDetector(
    SingleFamilyRestatedFactDetector, UndistinguishedTypeTestFindingRecipeSynthesizer
):
    """`isinstance` rungs that run the same code as their neighbour or as the else."""

    module_projection_family = UndistinguishedTypeArmsSiteFamily
    restated = RestatedFact(
        fact="The type tests tell these classes apart, then run the same code for each, so the ladder spells a distinction between them that nothing uses.",
        owner="The owner is one test: a single `isinstance` with every class whose arm is the same, or the else that already runs that code.",
        factoring="Merge adjacent rungs with the same body into one `isinstance(subject, (A, B))`, and delete trailing rungs whose body is the else body; rungs that are not adjacent stay, since moving a test above another would change which arm an instance of both classes takes.",
        operations=(MergeTypeTestArmsOperation,),
    )
    finding_spec = restated.spec(
        PatternId.UNDISTINGUISHED_TYPE_TEST,
        "Type tests tell classes apart that the code treats identically",
        "one type test per distinct arm",
        "an if/elif ladder whose every test is `isinstance` on one subject, with adjacent rungs running the same body, or trailing rungs running the else body",
        (),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def describe(self, sites: Sequence[RestatingSite], context: None) -> str:
        groups = _listed(
            f"line {site.lineno} on {', '.join(f'`{name}`' for name in site.detail)}"
            for site in sites
        )
        return f"`{sites[0].symbol}` tests classes whose arms run the same code: {groups}."


class NameOnlySubclassDetector(
    JoinedRestatedFactDetector[EmptySubclassIndex, RestatingSite],
    EmptySubclassFindingRecipeSynthesizer,
):
    """An empty subclass that every spelling only constructs or raises."""

    module_projection_families = (
        EmptySubclassSiteFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        EmptySubclassIndex.from_projection_groups
    )
    restated = RestatedFact(
        fact="The subclass declares nothing and nothing in the scan tells it apart from its base: every spelling of it constructs or raises it, so its name is the only difference.",
        owner="The owner is the base class, which already is everything the subclass is.",
        factoring="Delete the subclass and rewrite every import and reference onto the base; a reference that tests, catches, annotates or spells the subclass is a distinction, so the class stays.",
        operations=(RedirectEmptySubclassOperation,),
    )
    finding_spec = restated.spec(
        PatternId.NAME_ONLY_SUBCLASS,
        "A subclass differs from its base only by name",
        "the subclass deleted and its consumers using the base",
        "a top-level, undecorated, unregistered class with one concrete base and no body, whose name is unique in the scan and every spelling of which constructs or raises it",
        (),
        (ObservationTag.CLASS_FAMILY,),
    )

    def admits(self, sites: Sequence[RestatingSite], context: EmptySubclassIndex) -> bool:
        return sites[0].detail[0] in context.bases_by_symbol

    def describe(self, sites: Sequence[RestatingSite], context: EmptySubclassIndex) -> str:
        symbol, base = sites[0].detail
        siblings = context.siblings(symbol)
        named = ", ".join(f"`{name}`" for name in siblings)
        alongside = {
            0: "",
            1: f"; its sibling {named} differs the same way",
        }.get(len(siblings), f"; its siblings {named} differ the same way")
        return (
            f"`{sites[0].symbol}` differs from its base `{base}` only by name: "
            f"every spelling of it constructs or raises it{alongside}."
        )


class UndistinguishedEnumMembersDetector(
    JoinedRestatedFactDetector[EnumMemberIndex, RestatingSite],
    EnumFoldFindingRecipeSynthesizer,
):
    """Plain-Enum members that every spelling names together."""

    module_projection_families = (EnumUseSiteFamily, CompactModuleClassProjectionFamily)
    compact_shared_group_context_builder = staticmethod(
        EnumMemberIndex.from_projection_groups
    )
    restated = RestatedFact(
        fact="Every test that names one of these members names the others with it, so the enum declares a distinction between them that no code makes.",
        owner="The owner is one member: the first of them, which every grouped test already accepts.",
        factoring="Delete the other members and drop their spellings from each grouped test; a member read alone, by value, by name or by iterating the enum is a distinction, so the members stay.",
        operations=(FoldEnumMembersOperation,),
    )
    finding_spec = restated.spec(
        PatternId.UNDISTINGUISHED_ENUM_MEMBERS,
        "Enum members are only ever named together",
        "one member where the code never tells them apart",
        "members of a plain Enum with literal values, unique in the scan and spelled only by member access or as a type, whose every spelling is in `in (...)`, `case A | B` or `is A or is B` with all of them",
        (),
        (ObservationTag.CLOSED_FAMILY_CASES,),
    )

    def admits(self, sites: Sequence[RestatingSite], context: EnumMemberIndex) -> bool:
        return sites[0].detail[0] in context.folds

    def describe(self, sites: Sequence[RestatingSite], context: EnumMemberIndex) -> str:
        folds = "; ".join(
            ", ".join(f"`{sites[0].symbol}.{member}`" for member in members)
            for members in context.folds[sites[0].detail[0]]
        )
        return f"`{sites[0].symbol}` has members that every spelling names together: {folds}."


__all__ = (
    "IdenticalArmsDetector",
    "UndistinguishedEnumMembersDetector",
    "NameOnlySubclassDetector",
    "UndistinguishedTypeTestDetector",
)
