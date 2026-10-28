"""False distinctions: tests, classes and members that distinguish in name only.

Each detector here names code that tells cases apart and then treats them the
same, the fact that distinction restates, and the factoring that removes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..false_distinction_observations import IdenticalArmsSiteFamily
from ..patterns import PatternId
from ..restated_fact_observations import RestatingSite
from ..taxonomy import ObservationTag
from ._restated_facts import (
    RestatedFact,
    SingleFamilyRestatedFactDetector,
    _listed,
)


class IdenticalArmsDetector(SingleFamilyRestatedFactDetector):
    """A branch whose every arm is the same code."""

    module_projection_family = IdenticalArmsSiteFamily
    restated = RestatedFact(
        fact="Every arm of the branch runs the same code, so its tests distinguish cases that the code then treats identically; the distinction exists in the tests alone.",
        owner="The owner is the one arm: it runs whatever the tests say.",
        factoring="Replace the branch with its one arm and delete the tests; a test whose evaluation has an effect keeps that effect as its own statement.",
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


__all__ = ("IdenticalArmsDetector",)
