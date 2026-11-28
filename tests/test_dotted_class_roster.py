"""Regression tests for MEMB-2 dotted class-name roster detection, minimal sound
ancestor cover computation, and `RosterToCapabilityMixinOperation`."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.class_families.parsed_index import build_class_family_index
from nominal_refactor_advisor.codemod import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.detectors._base import DetectorConfig, IssueDetector
from nominal_refactor_advisor.detectors._dotted_class_roster import (
    DottedClassRosterDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.roster_to_capability_mixin import (
    RosterToCapabilityMixinOperation,
    minimal_sound_ancestor_cover,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _sample_tree(root: Path) -> None:
    """Mini `personal-model` fixture mirroring shared held-out bases + trainable parent trap."""
    _write(
        root,
        {
            "practice/__init__.py": "",
            "practice/env.py": (
                "class TaskKind:\n"
                '    """Base family."""\n'
            ),
            "practice/live_probe.py": (
                "class LiveProbeTaskMixin:\n"
                "    pass\n\n\n"
                "class GradedLadderProbe(LiveProbeTaskMixin):\n"
                '    """Shared base for held-out graded ladder probes."""\n'
            ),
            "practice/structural_tasks.py": (
                "from practice.env import TaskKind\n\n\n"
                "class StructuralKind(TaskKind):\n"
                "    pass\n\n\n"
                "class HeldOutStructuralKind(StructuralKind):\n"
                "    pass\n\n\n"
                "class HeldOutEconKind(HeldOutStructuralKind):\n"
                "    pass\n\n\n"
                "class HeldOutMathKind(HeldOutStructuralKind):\n"
                "    pass\n\n\n"
                "class TrainableEconKind(StructuralKind):\n"
                "    pass\n"
            ),
            "practice/ladder_tasks.py": (
                "from practice.env import TaskKind\n"
                "from practice.live_probe import GradedLadderProbe\n\n\n"
                "class HeldOutLadderA(TaskKind, GradedLadderProbe):\n"
                "    pass\n\n\n"
                "class HeldOutLadderB(TaskKind, GradedLadderProbe):\n"
                "    pass\n"
            ),
            "practice/visual_tasks.py": (
                "from practice.env import TaskKind\n\n\n"
                "class VisualEconomicsKind(TaskKind):\n"
                '    """Trainable concrete kind with a single held-out subclass."""\n\n\n'
                "class HeldOutVisualEconomicsKind(VisualEconomicsKind):\n"
                "    pass\n\n\n"
                "class GpqaHeldOut(TaskKind):\n"
                "    pass\n"
            ),
            "train_offbox/__init__.py": "",
            "train_offbox/guard.py": (
                "from practice.env import TaskKind\n"
                "from practice.visual_tasks import VisualEconomicsKind\n\n\n"
                "_HELDOUT_EVAL_ONLY_KINDS: frozenset[str] = frozenset({\n"
                '    "practice.structural_tasks.HeldOutEconKind",\n'
                '    "practice.structural_tasks.HeldOutMathKind",\n'
                '    "practice.ladder_tasks.HeldOutLadderA",\n'
                '    "practice.ladder_tasks.HeldOutLadderB",\n'
                '    "practice.visual_tasks.HeldOutVisualEconomicsKind",\n'
                '    "practice.visual_tasks.GpqaHeldOut",\n'
                "})\n\n\n"
                "def check_wired(wired: set[type[TaskKind]], all_kinds: list[type[TaskKind]]) -> list[type[TaskKind]]:\n"
                '    wired_qualnames = {f"{kind.__module__}.{kind.__name__}" for kind in wired}\n'
                "    assert not (wired_qualnames & _HELDOUT_EVAL_ONLY_KINDS)\n"
                "    return [\n"
                "        cls\n"
                "        for cls in all_kinds\n"
                '        if f"{cls.__module__}.{cls.__name__}" not in _HELDOUT_EVAL_ONLY_KINDS\n'
                "        and cls not in wired\n"
                "    ]\n"
            ),
        },
    )


def test_pattern_id_and_registries_include_dotted_class_roster() -> None:
    assert PatternId.DOTTED_CLASS_ROSTER == 64
    assert DottedClassRosterDetector in IssueDetector.__registry__.values()
    assert (
        RefactorRecipeOperation.__registry__["roster_to_capability_mixin"]
        is RosterToCapabilityMixinOperation
    )


def test_detector_flags_dotted_class_roster_and_computes_minimal_cover(
    tmp_path: Path,
) -> None:
    _sample_tree(tmp_path)
    with ScanCache.scope():
        modules = parse_python_modules(
            tmp_path,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=True),
        )
        findings = DottedClassRosterDetector().detect(modules, DetectorConfig())
    assert len(findings) == 1
    finding = findings[0]
    assert finding.pattern_id is PatternId.DOTTED_CLASS_ROSTER
    assert "6/6 dotted class paths" in finding.summary
    assert "under `practice.env.TaskKind`" in finding.summary
    # Cover = GradedLadderProbe + HeldOutStructuralKind + HeldOutVisualEconomicsKind + GpqaHeldOut = 4 (not 6!)
    assert "4 base declaration(s)" in finding.summary


def test_detector_ignores_non_class_dotted_strings(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/settings.py": (
                "SETTING_KEYS = frozenset({\n"
                '    "ui.theme.name",\n'
                '    "ui.sidebar.hide",\n'
                '    "tools.expand.mode",\n'
                "})\n"
            ),
        },
    )
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False)
        findings = DottedClassRosterDetector().detect(modules, DetectorConfig())
    assert findings == []


def test_minimal_sound_ancestor_cover_never_lifts_onto_concrete_trainable_parent(
    tmp_path: Path,
) -> None:
    _sample_tree(tmp_path)
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False)
        index = build_class_family_index(modules)
    targets = frozenset(
        {
            "practice.structural_tasks.HeldOutEconKind",
            "practice.structural_tasks.HeldOutMathKind",
            "practice.ladder_tasks.HeldOutLadderA",
            "practice.ladder_tasks.HeldOutLadderB",
            "practice.visual_tasks.HeldOutVisualEconomicsKind",
            "practice.visual_tasks.GpqaHeldOut",
        }
    )
    cover = minimal_sound_ancestor_cover(
        index,
        targets,
        family_root="practice.env.TaskKind",
    )
    assert set(cover) == {
        "practice.live_probe.GradedLadderProbe",
        "practice.structural_tasks.HeldOutStructuralKind",
        "practice.visual_tasks.HeldOutVisualEconomicsKind",
        "practice.visual_tasks.GpqaHeldOut",
    }
    # Never tags VisualEconomicsKind (trainable concrete parent!) or StructuralKind or LiveProbeTaskMixin.
    assert "practice.visual_tasks.VisualEconomicsKind" not in cover
    assert "practice.structural_tasks.StructuralKind" not in cover
    assert "practice.live_probe.LiveProbeTaskMixin" not in cover


def test_roster_to_capability_mixin_codemod_rewrites_cover_and_deletes_roster(
    tmp_path: Path,
) -> None:
    _sample_tree(tmp_path)
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False)
        snapshot = CodemodSourceSnapshot.from_modules(modules)
    guard_path = (tmp_path / "train_offbox/guard.py").as_posix()
    env_path = (tmp_path / "practice/env.py").as_posix()

    operation = RosterToCapabilityMixinOperation(
        target=SourceRewriteTarget(
            file_path=guard_path,
            qualname="_HELDOUT_EVAL_ONLY_KINDS",
        ),
        mixin_file_path=env_path,
        mixin_name="HeldOutEvalOnlyTaskKind",
        docstring="Capability mixin for strictly held-out evaluation-only TaskKinds.",
    )
    simulation = (
        RefactorRecipe("roster-to-mixin")
        .with_operation(operation)
        .simulate(snapshot)
    )
    rewritten = simulation.simulation.rewritten_sources

    # 1. Mixin declared cleanly in practice/env.py before TaskKind.
    env_src = rewritten[env_path]
    assert "class HeldOutEvalOnlyTaskKind:" in env_src
    assert env_src.index("class HeldOutEvalOnlyTaskKind:") < env_src.index("class TaskKind:")

    # 2. Cover classes receive absolute import + base-list entry; leaf subclasses under shared bases stay untouched.
    live_probe_src = rewritten[(tmp_path / "practice/live_probe.py").as_posix()]
    assert "from practice.env import HeldOutEvalOnlyTaskKind" in live_probe_src
    assert "class GradedLadderProbe(LiveProbeTaskMixin, HeldOutEvalOnlyTaskKind):" in live_probe_src

    structural_src = rewritten[(tmp_path / "practice/structural_tasks.py").as_posix()]
    assert "from practice.env import HeldOutEvalOnlyTaskKind" in structural_src
    assert "class HeldOutStructuralKind(StructuralKind, HeldOutEvalOnlyTaskKind):" in structural_src
    assert "class HeldOutEconKind(HeldOutStructuralKind):" in structural_src
    assert "class TrainableEconKind(StructuralKind):" in structural_src

    visual_src = rewritten[(tmp_path / "practice/visual_tasks.py").as_posix()]
    assert "class VisualEconomicsKind(TaskKind):" in visual_src
    assert "class HeldOutVisualEconomicsKind(VisualEconomicsKind, HeldOutEvalOnlyTaskKind):" in visual_src
    assert "class GpqaHeldOut(TaskKind, HeldOutEvalOnlyTaskKind):" in visual_src

    # ladder_tasks.py needs NO edits because GradedLadderProbe already covers both HeldOutLadderA/B!
    assert (tmp_path / "practice/ladder_tasks.py").as_posix() not in rewritten

    # 3. Roster deleted and membership checks rewritten to issubclass in train_offbox/guard.py.
    guard_src = rewritten[guard_path]
    assert "_HELDOUT_EVAL_ONLY_KINDS" not in guard_src
    assert "wired_qualnames" not in guard_src
    assert "from practice.env import HeldOutEvalOnlyTaskKind" in guard_src
    assert "assert not ({kind for kind in wired if issubclass(kind, HeldOutEvalOnlyTaskKind)})" in guard_src
    assert "if not issubclass(cls, HeldOutEvalOnlyTaskKind)" in guard_src

    # 4. Zero relative imports introduced across all rewritten files.
    for src in rewritten.values():
        assert "\nfrom ." not in f"\n{src}"


def test_roster_to_capability_mixin_fails_loud_on_stale_entry(tmp_path: Path) -> None:
    _sample_tree(tmp_path)
    guard = tmp_path / "train_offbox/guard.py"
    guard.write_text(
        guard.read_text().replace(
            '"practice.visual_tasks.GpqaHeldOut",',
            '"practice.visual_tasks.DeletedGhostKind",',
        ),
        encoding="utf-8",
    )
    with ScanCache.scope():
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path, use_parse_cache=False)
        )
    operation = RosterToCapabilityMixinOperation(
        target=SourceRewriteTarget(
            file_path=guard.as_posix(),
            qualname="_HELDOUT_EVAL_ONLY_KINDS",
        ),
        mixin_file_path=(tmp_path / "practice/env.py").as_posix(),
        mixin_name="HeldOutEvalOnlyTaskKind",
        docstring="Held-out capability.",
    )
    with pytest.raises(NeedsInput) as exc_info:
        operation.source_edits_from_snapshot(snapshot)
    assert exc_info.value.needs == (NeededInput.SOURCE_CHANGE,)
    assert "DeletedGhostKind" in str(exc_info.value)


def test_roster_to_capability_mixin_places_mixin_above_same_file_cover_class(
    tmp_path: Path,
) -> None:
    _sample_tree(tmp_path)
    guard_path = (tmp_path / "train_offbox/guard.py").as_posix()
    live_probe_path = (tmp_path / "practice/live_probe.py").as_posix()
    with ScanCache.scope():
        snapshot = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path, use_parse_cache=False)
        )
    operation = RosterToCapabilityMixinOperation(
        target=SourceRewriteTarget(
            file_path=guard_path,
            qualname="_HELDOUT_EVAL_ONLY_KINDS",
        ),
        mixin_file_path=live_probe_path,
        mixin_name="HeldOutEvalOnlyTaskKind",
        docstring="Capability mixin for held-out TaskKinds.",
    )
    simulation = (
        RefactorRecipe("same-file-mixin")
        .with_operation(operation)
        .simulate(snapshot)
    )
    live_probe_src = simulation.simulation.rewritten_sources[live_probe_path]
    assert "class HeldOutEvalOnlyTaskKind:" in live_probe_src
    assert "class GradedLadderProbe(LiveProbeTaskMixin, HeldOutEvalOnlyTaskKind):" in live_probe_src
    assert live_probe_src.index("class HeldOutEvalOnlyTaskKind:") < live_probe_src.index(
        "class GradedLadderProbe("
    )
