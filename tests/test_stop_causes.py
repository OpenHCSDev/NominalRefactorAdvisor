"""A run that stops without a proof names why on its stop line."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.codemod import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_runtime import (
    FindingRecipeTrajectoryObstacle,
    FindingRecipeTrajectoryObstacleKind,
)
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodRefactorRecipeFrontierObstacle,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.refactor_concepts import PackageMoveConcept
from tests.test_family_walk_root import _family
from tests.test_package_move_batch import _write


def test_a_refused_move_names_the_refusal_on_the_stop_line(tmp_path: Path) -> None:
    """The only row refuses, so the run stops at a dead end. The stop line says
    the refusal itself, not just "no proved trajectory"."""

    _write(tmp_path, _family(""))
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(package_destination_tables=("| `messaging` | `alpha.py` | |\n",)),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    stop_line = report.to_markdown().splitlines()[2]
    assert stop_line.startswith("   - Stop reason: no_proved_trajectory; refused: Alpha is a member")
    refusal = (
        "refused: Alpha is a member of the Plugin family, and moving it to "
        "pkg.messaging.alpha would stop it registering: it walks pkg's direct "
        "modules (the walk is flat), and pkg.messaging.alpha is outside it"
    )
    assert report.stop_causes == (refusal,)


def test_an_incomplete_frontier_is_caused_by_its_refused_recipes() -> None:
    """"The recipe frontier is incomplete" names no cause; the recipes it refused do."""

    conflict = "recipe set has conflicting source edits: A deleted source cannot also be rewritten"
    obstacle = CodemodRefactorRecipeFrontierObstacle(
        source_state_id="state",
        depth=0,
        recipe_obstacles=(
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.CANDIDATE_SIMULATION,
                finding_ids=("a",),
                reason=conflict,
            ),
            FindingRecipeTrajectoryObstacle(
                kind=FindingRecipeTrajectoryObstacleKind.CANDIDATE_SIMULATION,
                finding_ids=("b",),
                reason=conflict,
            ),
        ),
    )
    assert obstacle.causes == (conflict,)
