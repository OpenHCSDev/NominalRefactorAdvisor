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
