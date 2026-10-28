"""False distinctions: tests, classes and members that distinguish in name only.

Each detector here names code that tells cases apart and then treats them the
same, the fact that distinction restates, and the factoring that removes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..codemod_false_distinctions import (
    CollapseIdenticalArmsOperation,
    IdenticalArmsFindingRecipeSynthesizer,
    MergeTypeTestArmsOperation,
    UndistinguishedTypeTestFindingRecipeSynthesizer,
)
from ..false_distinction_observations import (
    IdenticalArmsSiteFamily,
    UndistinguishedTypeArmsSiteFamily,
)
from ..patterns import PatternId
from ..restated_fact_observations import RestatingSite
from ..taxonomy import ObservationTag
from ._restated_facts import (
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


__all__ = ("IdenticalArmsDetector", "UndistinguishedTypeTestDetector")
