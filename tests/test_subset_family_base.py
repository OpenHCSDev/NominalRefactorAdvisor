"""A roster of some family members becomes a declared base in one proved move."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._semantic_descent import SemanticMirrorWithoutDescentDetector
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_candidates import OwnerPick
from nominal_refactor_advisor.owner_picks import OwnerPicks
from nominal_refactor_advisor.refactors.subset_family_base import (
    DeclareSubsetFamilyBaseOperation,
)

_HOOK = (
    "class Phase:\n"
    "    registry: dict = {}\n"
    "    after: tuple = ()\n\n"
    "    def __init_subclass__(cls, declared_name: str | None = '', **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        if declared_name is not None:\n"
    "            Phase.registry[declared_name or cls.__name__] = cls\n\n"
)

_BUS = _HOOK + (
    "class VirtualBusPhase(Phase, declared_name=None):\n"
    '    """The bus plan."""\n\n'
    "class BusInterfacePhase(VirtualBusPhase):\n"
    "    evidence = 'interface'\n\n"
    "class BusWriterPhase(VirtualBusPhase):\n"
    "    evidence = 'writer'\n"
    "    after = (BusInterfacePhase,)\n\n"
    "class BusDeliveryReadersPhase(VirtualBusPhase):\n"
    '    """Delivery readers."""\n\n'
    "    after = (BusWriterPhase,)\n"
    "    evidence = 'delivery'\n\n"
    "class BusHistoryReadersPhase(VirtualBusPhase):\n"
    "    after = (BusWriterPhase,)\n"
    "    evidence = 'history'\n\n"
    "class BusCarryPhase(VirtualBusPhase):\n"
    "    after = (BusWriterPhase,)\n\n"
    "class BusWatchedPhase(VirtualBusPhase):\n"
    "    after = (BusWriterPhase,)\n"
    "    evidence = 'watched'\n\n"
    "class BusDeletionPhase(VirtualBusPhase):\n"
    "    after = (BusDeliveryReadersPhase, BusHistoryReadersPhase, BusCarryPhase)\n"
)

_TEAM = _HOOK + (
    "class TeamAwarenessPhase(Phase, declared_name=None):\n"
    '    """Fleet awareness."""\n\n'
    "class OneHolderPhase(TeamAwarenessPhase):\n"
    "    evidence = 'holder'\n\n"
    "class DriftPhase(TeamAwarenessPhase):\n"
    "    evidence = 'drift'\n\n"
    "class SpinningPhase(TeamAwarenessPhase, declared_name='spinning_unit'):\n"
    "    evidence = 'spinning'\n\n"
    "class TeamReplayPhase(TeamAwarenessPhase, declared_name='fleet_replay'):\n"
    "    after = (\n"
    "        OneHolderPhase,\n"
    "        DriftPhase,\n"
    "        SpinningPhase,\n"
    "    )\n"
)


def _operation(
    tmp_path: Path, source: str, authority: str, holder: str, name: str
) -> tuple[CodemodSourceSnapshot, DeclareSubsetFamilyBaseOperation, Path]:
    path = tmp_path / "pkg" / "phases.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    targets = {target.qualname: target for target in snapshot.source_index.ast_targets}
    operation = DeclareSubsetFamilyBaseOperation(
        target=SourceRewriteTarget(target_id=targets[authority].target_id),
        projection_target=SourceRewriteTarget(target_id=targets[holder].target_id),
        base_name=name,
        docstring="The phases that read what the writer wrote.",
    )
    return snapshot, operation, path


def _rewritten(snapshot: CodemodSourceSnapshot, operation, path: Path) -> str:
    simulation = RefactorRecipe("declare").with_operation(operation).simulate(snapshot)
    return simulation.simulation.rewritten_sources[path.as_posix()]


def _run(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, "phases", "exec"), namespace)
    return namespace


def test_the_bus_readers_become_a_base_and_the_deletion_reads_it(tmp_path: Path) -> None:
    """plan_phases.py:306: three readers restate `after = (BusWriterPhase,)`; the
    watched phase restates it too but is outside the roster and keeps its own."""

    snapshot, operation, path = _operation(
        tmp_path, _BUS, "VirtualBusPhase", "BusDeletionPhase", "BusReaderPhase"
    )
    rewritten = _rewritten(snapshot, operation, path)

    assert (
        "class BusReaderPhase(VirtualBusPhase, declared_name=None):\n"
        '    """The phases that read what the writer wrote."""\n\n'
        "    after = (BusWriterPhase,)\n\n\n"
        "class BusDeliveryReadersPhase(BusReaderPhase):\n"
    ) in rewritten
    assert "class BusCarryPhase(BusReaderPhase):\n    pass\n" in rewritten
    assert "class BusWatchedPhase(VirtualBusPhase):\n    after = (BusWriterPhase,)\n" in rewritten
    assert "    after = tuple(BusReaderPhase.__subclasses__())\n" in rewritten
    assert rewritten.count("after = (BusWriterPhase,)") == 2

    before, after = _run(_BUS), _run(rewritten)
    assert [c.__name__ for c in after["BusDeletionPhase"].after] == [
        c.__name__ for c in before["BusDeletionPhase"].after
    ]
    for name in ("BusDeliveryReadersPhase", "BusHistoryReadersPhase", "BusCarryPhase"):
        assert after[name].after == (after["BusWriterPhase"],)
        assert getattr(after[name], "evidence", None) == getattr(before[name], "evidence", None)
    assert sorted(after["Phase"].registry) == sorted(before["Phase"].registry)


def test_the_incident_phases_become_a_base_the_replay_reads(tmp_path: Path) -> None:
    """agent-comms #4853: the replay is itself an awareness phase, so its roster of
    every other one is a subset: the incidents."""

    snapshot, operation, path = _operation(
        tmp_path, _TEAM, "TeamAwarenessPhase", "TeamReplayPhase", "TeamIncidentPhase"
    )
    rewritten = _rewritten(snapshot, operation, path)

    assert "class TeamIncidentPhase(TeamAwarenessPhase, declared_name=None):\n" in rewritten
    assert "class SpinningPhase(TeamIncidentPhase, declared_name='spinning_unit'):\n" in rewritten
    assert (
        "class TeamReplayPhase(TeamAwarenessPhase, declared_name='fleet_replay'):\n"
        "    after = tuple(TeamIncidentPhase.__subclasses__())\n"
    ) in rewritten

    before, after = _run(_TEAM), _run(rewritten)
    assert [c.__name__ for c in after["TeamReplayPhase"].after] == [
        c.__name__ for c in before["TeamReplayPhase"].after
    ]
    assert sorted(after["Phase"].registry) == sorted(before["Phase"].registry)
    assert "TeamIncidentPhase" not in after["Phase"].registry


def test_a_roster_out_of_declaration_order_asks_for_the_order(tmp_path: Path) -> None:
    source = _BUS.replace(
        "(BusDeliveryReadersPhase, BusHistoryReadersPhase, BusCarryPhase)",
        "(BusCarryPhase, BusDeliveryReadersPhase, BusHistoryReadersPhase)",
    )
    snapshot, operation, _ = _operation(
        tmp_path, source, "VirtualBusPhase", "BusDeletionPhase", "BusReaderPhase"
    )
    with pytest.raises(NeedsInput, match="out of declaration order") as refusal:
        operation.source_edits(snapshot)
    assert refusal.value.needs == (NeededInput.SOURCE_CHANGE,)


def test_an_unordered_roster_takes_the_base_in_any_order(tmp_path: Path) -> None:
    source = _BUS.replace(
        "(BusDeliveryReadersPhase, BusHistoryReadersPhase, BusCarryPhase)",
        "frozenset({BusCarryPhase, BusDeliveryReadersPhase, BusHistoryReadersPhase})",
    )
    snapshot, operation, path = _operation(
        tmp_path, source, "VirtualBusPhase", "BusDeletionPhase", "BusReaderPhase"
    )
    assert "    after = frozenset(BusReaderPhase.__subclasses__())\n" in _rewritten(
        snapshot, operation, path
    )


def test_a_read_of_the_authoritys_direct_members_keeps_them(tmp_path: Path) -> None:
    source = _BUS + "\nBUS = tuple(VirtualBusPhase.__subclasses__())\n"
    snapshot, operation, _ = _operation(
        tmp_path, source, "VirtualBusPhase", "BusDeletionPhase", "BusReaderPhase"
    )
    with pytest.raises(NeedsInput, match="leave it"):
        operation.source_edits(snapshot)


def test_the_base_name_is_a_pick(tmp_path: Path) -> None:
    snapshot, operation, _ = _operation(
        tmp_path, _BUS, "VirtualBusPhase", "BusDeletionPhase", "BusWriterPhase"
    )
    with pytest.raises(NeedsInput, match="already bound") as refusal:
        operation.source_edits(snapshot)
    assert refusal.value.needs == (NeededInput.OWNER_PICK,)


def test_a_picked_base_reaches_the_plan_as_its_recipe(tmp_path: Path) -> None:
    """The pick names the family and the base; the plan offers the declared base in
    place of deriving the roster from the whole family."""

    path = tmp_path / "pkg" / "phases.py"
    path.parent.mkdir(parents=True)
    path.write_text(_BUS, encoding="utf-8")
    modules = parse_python_modules(tmp_path)
    finding = next(
        finding
        for finding in SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
        if finding.authority_evidence is not None
        and finding.authority_evidence.symbol == "VirtualBusPhase"
    )
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="VirtualBusPhase",
                because="the readers share their writer",
                name="BusReaderPhase",
                judgement="The phases that read what the writer wrote.",
            ),
        ),
    )
    picked, report = picks.applied((finding,))
    assert [outcome.outcome for outcome in report.outcomes] == ["picked"]
    assert report.unmatched == ()

    plan = codemod_plan_from_findings(
        tuple(picked), selector_context=CodemodSourceSnapshot.from_modules(modules, tuple(picked))
    )
    (recipe,) = plan.document.recipes
    assert recipe.recipe_id.endswith("-declare-subset-family-base")
    assert [type(operation) for operation in recipe.operations] == [
        DeclareSubsetFamilyBaseOperation
    ]


def test_a_pick_below_another_family_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pkg" / "phases.py"
    path.parent.mkdir(parents=True)
    path.write_text(_BUS, encoding="utf-8")
    finding = next(
        finding
        for finding in SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path), DetectorConfig()
        )
        if finding.authority_evidence is not None
        and finding.authority_evidence.symbol == "VirtualBusPhase"
    )
    pick = OwnerPick(finding.stable_id, "Phase", "", "BusReaderPhase", "Readers.")
    picked, report = OwnerPicks("picks.json", "", (pick,)).applied((finding,))
    assert [outcome.outcome for outcome in report.outcomes] == ["rejected"]
    assert picked == [finding]
