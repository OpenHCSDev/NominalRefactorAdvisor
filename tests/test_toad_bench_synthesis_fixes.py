"""Regression fixtures lifted from Toad bench synthesis failures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import ConstantSubtypeAnswerDetector
from nominal_refactor_advisor.detectors._restated_facts import (
    ForeignAbsenceProbeDetector,
    OwnTypeTestDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _parse(root: Path):
    return parse_python_modules(
        root,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )


def test_constant_subtype_compound_and_conditional_return_before_tail_pushes(
    tmp_path: Path,
) -> None:
    """`if head.draws and enabled: ... if cond: return fast` followed by a terminating
    tail pushes cleanly onto the family (Toad `transcript_history.py:719` pattern)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/placement.py": (
                "class HeadPlacement:\n"
                "    draws = False\n\n\n"
                "class DrawnHead(HeadPlacement):\n"
                "    draws = True\n\n\n"
                "class UndrawnHead(HeadPlacement):\n"
                "    draws = False\n\n\n"
                "def reveal_head(placement: HeadPlacement, enabled: bool, target_y: int, scroll_y: int) -> str:\n"
                "    if placement.draws and enabled:\n"
                "        if target_y < scroll_y:\n"
                "            return f'glide:{target_y}'\n"
                "    return f'jump:{scroll_y}'\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    with ScanCache.scope():
        (finding,) = ConstantSubtypeAnswerDetector().detect(modules, DetectorConfig())
    snap = CodemodSourceSnapshot.from_modules(modules)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, snap
    )
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()

    # Verify rewritten module clears finding and preserves exact runtime semantics.
    with ScanCache.scope():
        assert ConstantSubtypeAnswerDetector().detect(_parse(tmp_path), DetectorConfig()) == []
    spec = importlib.util.spec_from_file_location("pkg_placement", tmp_path / "pkg/placement.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.reveal_head(mod.DrawnHead(), True, 10, 50) == "glide:10"
    assert mod.reveal_head(mod.DrawnHead(), True, 60, 50) == "jump:50"
    assert mod.reveal_head(mod.DrawnHead(), False, 10, 50) == "jump:50"
    assert mod.reveal_head(mod.UndrawnHead(), True, 10, 50) == "jump:50"


def test_foreign_absence_probe_non_ast_external_owner_does_not_claim_missing_ast_decoder(
    tmp_path: Path,
) -> None:
    """Probing an external non-AST class or unresolved receiver reports SOURCE_CHANGE
    naming the owner instead of blaming missing OptionalAstFieldDecoder entries."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/consumer.py": (
                "from pathlib import Path\n\n\n"
                "def probe_external(p: Path) -> str:\n"
                "    if p.parent is not None:\n"
                "        return str(p.parent)\n"
                "    return ''\n\n\n"
                "def probe_unresolved(row) -> bool:\n"
                "    return False if row.wrap_mode is None else True\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        findings = ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
    assert len(findings) == 2
    for finding in findings:
        evaluation = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
            finding, snap
        )
        assert "registered decoders cover" not in evaluation.rejection_reason
        assert evaluation.needs == (NeededInput.SOURCE_CHANGE,)


def test_own_type_test_resolves_subscripted_tuple_element_annotation(
    tmp_path: Path,
) -> None:
    """`isinstance(blocks[index], Divider)` where `blocks: tuple[LineBlock, ...]`
    resolves `LineBlock` from the declared element annotation (Toad `qt/history.py:1733`)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/blocks.py": (
                "class LineBlock:\n"
                "    pass\n\n\n"
                "class Divider(LineBlock):\n"
                "    pass\n\n\n"
                "class Body(LineBlock):\n"
                "    pass\n\n\n"
                "def header_collapse_at(blocks: tuple[LineBlock, ...], block_index: int) -> bool:\n"
                "    if isinstance(blocks[block_index], Divider):\n"
                "        return True\n"
                "    return False\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        (finding,) = OwnTypeTestDetector().detect(modules, DetectorConfig())
    evaluation = OwnTypeTestDetector().evaluate_recipe_for_finding(finding, snap)
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()

    with ScanCache.scope():
        assert OwnTypeTestDetector().detect(_parse(tmp_path), DetectorConfig()) == []
    spec = importlib.util.spec_from_file_location("pkg_blocks", tmp_path / "pkg/blocks.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blocks = (mod.Divider(), mod.Body())
    assert mod.header_collapse_at(blocks, 0) is True
    assert mod.header_collapse_at(blocks, 1) is False


def test_foreign_absence_probe_unresolved_ast_colliding_field_in_non_ast_module(
    tmp_path: Path,
) -> None:
    """Probing `.value` or `.name` on an unresolved `Any` receiver in a module that
    does not import `ast` reports SOURCE_CHANGE instead of an OptionalValueDecoder /
    CaptureNameDecoder handback (Toad `model_selection.py:215`, `interactive_dialog.py:236`)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/ui.py": (
                "def selected(choice, event) -> tuple[bool, bool]:\n"
                "    return (choice.value is not None, event.pressed.name is not None)\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        findings = ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
    assert len(findings) == 2
    for finding in findings:
        evaluation = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
            finding, snap
        )
        assert "registered decoders cover" not in evaluation.rejection_reason
        assert "handback" not in evaluation.rejection_reason
        assert evaluation.needs == (NeededInput.SOURCE_CHANGE,)


def test_foreign_absence_prefix_guarded_or_ternary_with_local_name_fallback_pushes(
    tmp_path: Path,
) -> None:
    """`return width if entry is None or entry.extent is None else entry.extent`
    pushes onto `RowMetrics` cleanly (Toad `qt/virtual_scroll.py:555` pattern)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/metrics.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass\n"
                "class RowMetrics:\n"
                "    extent: int | None = None\n\n\n"
                "def row_extent(table: dict[int, RowMetrics], width: int) -> int:\n"
                "    entry = table.get(width)\n"
                "    return width if entry is None or entry.extent is None else entry.extent\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        (finding,) = ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding.stable_id,
                "row_extent",
                "RowMetrics owns its effective extent.",
                "extent_or",
                "Return recorded extent or fallback width.",
            ),
        ),
    )
    (picked,), report = picks.applied((finding,))
    assert not report.unmatched
    evaluation = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(picked, snap)
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()

    with ScanCache.scope():
        assert ForeignAbsenceProbeDetector().detect(_parse(tmp_path), DetectorConfig()) == []
    spec = importlib.util.spec_from_file_location("pkg_metrics", tmp_path / "pkg/metrics.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.row_extent({}, 80) == 80
    assert mod.row_extent({80: mod.RowMetrics(None)}, 80) == 80
    assert mod.row_extent({80: mod.RowMetrics(120)}, 80) == 120


def test_constant_subtype_compound_and_uniform_early_return_without_tail_pushes(
    tmp_path: Path,
) -> None:
    """`if placed.located and placed.ready(w): ... if cond: return head` at the end
    of an outer `if` (empty local tail) pushes onto the family via `if placed.m(...): return head`
    (Toad `transcript_history.py:701` pattern)."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/placed.py": (
                "class FragmentLocation:\n"
                "    located = True\n"
                "    def ready(self, limit: int) -> bool:\n"
                "        return limit > 0\n"
                "    def target_y(self) -> int:\n"
                "        return 10\n\n\n"
                "class UnloadedFragment(FragmentLocation):\n"
                "    located = False\n\n\n"
                "def maybe_reveal(placed: FragmentLocation, enabled: bool, limit: int, scroll_y: int, head: str) -> str:\n"
                "    if enabled:\n"
                "        if placed.located and placed.ready(limit):\n"
                "            y = placed.target_y()\n"
                "            if y < scroll_y:\n"
                "                return head\n"
                "    return 'fallback'\n"
            ),
        },
    )
    modules = _parse(tmp_path)
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        (finding,) = ConstantSubtypeAnswerDetector().detect(modules, DetectorConfig())
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding.stable_id,
                "maybe_reveal",
                "FragmentLocation owns reveal eligibility.",
                "can_reveal",
                "Whether this placed location can reveal above scroll_y.",
            ),
        ),
    )
    (picked,), report = picks.applied((finding,))
    assert not report.unmatched
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(picked, snap)
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()

    with ScanCache.scope():
        assert ConstantSubtypeAnswerDetector().detect(_parse(tmp_path), DetectorConfig()) == []
    spec = importlib.util.spec_from_file_location("pkg_placed", tmp_path / "pkg/placed.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.maybe_reveal(mod.FragmentLocation(), True, 5, 20, "H") == "H"
    assert mod.maybe_reveal(mod.FragmentLocation(), True, 5, 5, "H") == "fallback"
    assert mod.maybe_reveal(mod.FragmentLocation(), True, 0, 20, "H") == "fallback"
    assert mod.maybe_reveal(mod.UnloadedFragment(), True, 5, 20, "H") == "fallback"
    assert mod.maybe_reveal(mod.FragmentLocation(), False, 5, 20, "H") == "fallback"
