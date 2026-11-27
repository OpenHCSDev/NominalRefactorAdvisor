"""Tests for Pattern 44 (`threaded_absence`) Held/Absent nominal family codemod."""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import codemod_plan_from_findings
from nominal_refactor_advisor.codemod_runtime import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import ThreadedAbsenceDetector
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _scan(root: Path):
    modules = parse_python_modules(
        root,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        findings = ThreadedAbsenceDetector().detect(modules, DetectorConfig())
    return modules, snap, findings


def test_threaded_absence_void_helper_scaffolds_noop_base_and_rewrites_callers(
    tmp_path: Path,
) -> None:
    """A private void helper probing `gallery_dir is not None` scaffolds a base with a
    default no-op method, `Absent<Family>`, `Held<Family>(value: T)`, and `from_optional`,
    and rewrites literal `None`, non-None, ternary, and dynamic callers."""

    source = (
        "from __future__ import annotations\n\n\n"
        "def _record_tag(gallery_dir: str | None, step: int, log: list[str]) -> None:\n"
        "    if gallery_dir is not None:\n"
        "        log.append(f'{gallery_dir}/step_{step}')\n\n\n"
        "def run_all(\n"
        "    step: int, save: bool, root: str, maybe: str | None, log: list[str]\n"
        ") -> None:\n"
        "    _record_tag(None, step, log)\n"
        "    _record_tag(root, step, log)\n"
        "    _record_tag(root if save else None, step, log)\n"
        "    _record_tag(maybe, step, log)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/gallery.py": source})
    _modules, snap, findings = _scan(tmp_path)
    assert len(findings) == 1
    detector = ThreadedAbsenceDetector()

    unpicked = detector.evaluate_recipe_for_finding(findings[0], snap)
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=findings[0].stable_id,
                owner="_record_tag",
                because="gallery dir absence belongs on a nominal target family",
                name="GalleryTarget",
                judgement="Nominal gallery export target deciding whether to record a step.",
                hoisted_name="record_step",
                hoisted_judgement="Record this step when a gallery directory is held.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report

    replay = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).simulate(snap, backend=CodemodBackend.AST_SPAN)
    rewritten = sim.simulation.rewritten_sources[str(tmp_path / "pkg/gallery.py")]
    assert (
        replay.simulation.rewritten_sources[str(tmp_path / "pkg/gallery.py")]
        == rewritten
    )
    sim.apply()

    assert "class GalleryTarget(ABC):" in rewritten
    assert "class AbsentGalleryTarget(GalleryTarget):" in rewritten
    assert "class HeldGalleryTarget(GalleryTarget):" in rewritten
    assert "def from_optional(cls, value: str | None) -> GalleryTarget:" in rewritten
    assert "_record_tag(AbsentGalleryTarget(), step, log)" in rewritten
    assert "_record_tag(HeldGalleryTarget(root), step, log)" in rewritten
    assert (
        "_record_tag(HeldGalleryTarget(root) if save else AbsentGalleryTarget(), step, log)"
        in rewritten
    )
    assert "_record_tag(GalleryTarget.from_optional(maybe), step, log)" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "gallery.py", "exec"), ns)
    log1: list[str] = []
    ns["run_all"](7, True, "/g", "/m", log1)  # type: ignore[operator]
    assert log1 == ["/g/step_7", "/g/step_7", "/m/step_7"]
    log2: list[str] = []
    ns["run_all"](9, False, "/g", None, log2)  # type: ignore[operator]
    assert log2 == ["/g/step_9"]

    _, _, after_findings = _scan(tmp_path)
    assert after_findings == []


def test_threaded_absence_returning_method_cross_module_callers_and_embedded_ternary(
    tmp_path: Path,
) -> None:
    """A staticmethod with early return (`if max_tokens is None: return True; ...`)
    and a cross-module caller plus an embedded ternary branch both rewrite cleanly
    with `@abstractmethod`, cross-module imports, and exact runtime parity."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/budget.py": (
                "from __future__ import annotations\n\n"
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Row:\n"
                "    tokens: int\n\n\n"
                "class Attempt:\n"
                "    @staticmethod\n"
                "    def _prompt_fits(row: Row, max_tokens: int | None) -> bool:\n"
                "        if max_tokens is None:\n"
                "            return True\n"
                "        return row.tokens < max_tokens\n\n"
                "    def keep_unbounded(self, row: Row) -> bool:\n"
                "        return self._prompt_fits(row, None)\n\n\n"
                "def _announce_verb(previous: str | None) -> str:\n"
                "    verb = 'Hot-swapped' if previous is not None else 'Serving'\n"
                "    return f'[{verb}]'\n\n\n"
                "def format_initial() -> str:\n"
                "    return _announce_verb(None)\n\n\n"
                "def format_swap(prev: str) -> str:\n"
                "    return _announce_verb(prev)\n"
            ),
            "pkg/caller.py": (
                "from __future__ import annotations\n\n"
                "from pkg.budget import Attempt, Row\n\n\n"
                "def check_bounded(tokens: int, limit: int) -> bool:\n"
                "    return Attempt._prompt_fits(Row(tokens), limit)\n"
            ),
        },
    )

    _modules, snap, findings = _scan(tmp_path)
    by_sym = {f.evidence[0].symbol: f for f in findings}
    assert set(by_sym) == {"Attempt._prompt_fits", "_announce_verb"}

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=by_sym["Attempt._prompt_fits"].stable_id,
                owner="Attempt._prompt_fits",
                because="prompt token limit belongs on a nominal budget family",
                name="PromptTokenBudget",
                judgement="Nominal prompt token budget deciding whether a row fits.",
                hoisted_name="fits_row",
            ),
            OwnerPick(
                finding=by_sym["_announce_verb"].stable_id,
                owner="_announce_verb",
                because="previous revision absence belongs on a nominal prior-revision family",
                name="PriorRevision",
                judgement="Nominal prior revision deciding announcement phrasing.",
                hoisted_name="announce_verb",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 2
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    sim.apply()

    budget_after = (tmp_path / "pkg/budget.py").read_text()
    caller_after = (tmp_path / "pkg/caller.py").read_text()
    assert "@abstractmethod" in budget_after
    assert "from_optional" not in budget_after
    assert "class PromptTokenBudget(ABC):" in budget_after
    assert "class PriorRevision(ABC):" in budget_after
    assert "HeldPromptTokenBudget" in caller_after
    assert "Attempt._prompt_fits(Row(tokens), HeldPromptTokenBudget(limit))" in caller_after

    budget_ns: dict[str, object] = {}
    exec(compile(budget_after, "budget.py", "exec"), budget_ns)
    assert budget_ns["Attempt"]().keep_unbounded(budget_ns["Row"](500)) is True  # type: ignore[operator]
    assert budget_ns["format_initial"]() == "[Serving]"  # type: ignore[operator]
    assert budget_ns["format_swap"]("v1") == "[Hot-swapped]"  # type: ignore[operator]

    caller_ns: dict[str, object] = {
        "Attempt": budget_ns["Attempt"],
        "Row": budget_ns["Row"],
        "HeldPromptTokenBudget": budget_ns["HeldPromptTokenBudget"],
    }
    caller_exec = "\n".join(
        line
        for line in caller_after.splitlines()
        if not line.startswith("from pkg.budget import")
    )
    exec(compile(caller_exec, "caller.py", "exec"), caller_ns)
    assert caller_ns["check_bounded"](10, 20) is True  # type: ignore[operator]
    assert caller_ns["check_bounded"](25, 20) is False  # type: ignore[operator]

    _, _, after_findings = _scan(tmp_path)
    assert after_findings == []


def test_threaded_absence_refuses_unsound_callers_and_escaped_parameter(
    tmp_path: Path,
) -> None:
    """Parameters still referenced outside the pushed branch, functions passed as
    first-class values, `*args` splats, `__all__` exports, and overridden methods
    must all stay soundly refused."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/unsafe.py": (
                "from __future__ import annotations\n\n"
                "__all__ = ['exported_fn']\n\n\n"
                "def _still_used(x: str | None) -> str | None:\n"
                "    if x is not None:\n"
                "        _ = len(x)\n"
                "    return x\n\n\n"
                "def _callback_fn(x: str | None) -> int:\n"
                "    if x is None:\n"
                "        return 0\n"
                "    return len(x)\n\n\n"
                "def _splatted(x: str | None) -> int:\n"
                "    if x is None:\n"
                "        return 0\n"
                "    return len(x)\n\n\n"
                "def exported_fn(x: str | None) -> int:\n"
                "    if x is None:\n"
                "        return 0\n"
                "    return len(x)\n\n\n"
                "class Base:\n"
                "    def _overridden(self, x: str | None) -> int:\n"
                "        if x is None:\n"
                "            return 0\n"
                "        return len(x)\n\n\n"
                "class Child(Base):\n"
                "    def _overridden(self, x: str | None) -> int:\n"
                "        return 1\n\n\n"
                "def use_them(args: tuple[str | None, ...]) -> None:\n"
                "    _still_used(None)\n"
                "    cb = _callback_fn\n"
                "    _ = cb(None)\n"
                "    _splatted(*args)\n"
                "    exported_fn(None)\n"
                "    Base()._overridden(None)\n"
            ),
        },
    )

    _modules, snap, findings = _scan(tmp_path)
    by_sym = {f.evidence[0].symbol: f for f in findings}
    detector = ThreadedAbsenceDetector()

    for sym, expected_substr in (
        ("_still_used", "still referenced outside"),
        ("_callback_fn", "references `_callback_fn` as a value"),
        ("_splatted", "uses `*args` or `**kwargs`"),
        ("exported_fn", "listed in `__all__`"),
        ("Base._overridden", "also declares `_overridden`"),
    ):
        ev = detector.evaluate_recipe_for_finding(by_sym[sym], snap)
        assert ev.recipe is None
        assert NeededInput.OWNER_PICK not in ev.needs, (sym, ev.reason)
        assert expected_substr in ev.reason, (sym, ev.reason)


def test_threaded_absence_compound_boolop_returns_ifs_ifexp_and_comprehension(
    tmp_path: Path,
) -> None:
    """Compound `BoolOp` shapes (`return param is None or ...`, `return bool(param is not None and ...)`,
    `if param is None or ...: return ...`, `if param is not None and ...: ... else: ...`,
    `A if param is None or ... else B`, and comprehension filter) all push cleanly with runtime parity."""

    source = (
        "from __future__ import annotations\n\n\n"
        "def _matches(target: str | None, items: tuple[str, ...]) -> bool:\n"
        "    return target is None or target in items\n\n\n"
        "def _has_prefix(prefix: str | None, text: str) -> bool:\n"
        "    return bool(prefix is not None and text.startswith(prefix))\n\n\n"
        "def _advance(current: str | None, blocked: frozenset[str]) -> str:\n"
        "    if current is None or current in blocked:\n"
        "        return 'stop'\n"
        "    return f'go:{current}'\n\n\n"
        "def _pick_mode(tag: str | None, strict: bool) -> str:\n"
        "    if tag is not None and (strict or len(tag) > 2):\n"
        "        return f'tagged:{tag}'\n"
        "    else:\n"
        "        return 'plain'\n\n\n"
        "def _label(opt: str | None) -> str:\n"
        "    return 'fast' if opt is None or opt == 'auto' else f'custom:{opt}'\n\n\n"
        "def _filter_items(cut: int | None, values: tuple[int, ...]) -> tuple[int, ...]:\n"
        "    return tuple(v for v in values if cut is None or v <= cut)\n\n\n"
        "def run_suite() -> tuple[object, ...]:\n"
        "    return (\n"
        "        _matches(None, ('a',)),\n"
        "        _matches('a', ('a',)),\n"
        "        _matches('b', ('a',)),\n"
        "        _has_prefix(None, 'abc'),\n"
        "        _has_prefix('ab', 'abc'),\n"
        "        _has_prefix('zz', 'abc'),\n"
        "        _advance(None, frozenset({'x'})),\n"
        "        _advance('x', frozenset({'x'})),\n"
        "        _advance('y', frozenset({'x'})),\n"
        "        _pick_mode(None, True),\n"
        "        _pick_mode('a', False),\n"
        "        _pick_mode('a', True),\n"
        "        _pick_mode('long', False),\n"
        "        _label(None),\n"
        "        _label('auto'),\n"
        "        _label('gpu'),\n"
        "        _filter_items(None, (1, 5, 9)),\n"
        "        _filter_items(5, (1, 5, 9)),\n"
        "    )\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/compound.py": source})

    ns_before: dict[str, object] = {}
    exec(compile(source, "compound.py", "exec"), ns_before)
    expected = ns_before["run_suite"]()  # type: ignore[operator]

    _, snap, findings = _scan(tmp_path)
    by_sym = {f.evidence[0].symbol: f for f in findings}
    assert set(by_sym) == {
        "_matches",
        "_has_prefix",
        "_advance",
        "_pick_mode",
        "_label",
        "_filter_items",
    }

    pick_specs = (
        ("_matches", "MatchTarget", "Nominal match target.", None),
        ("_has_prefix", "PrefixFilter", "Nominal prefix filter.", None),
        ("_advance", "StepCursor", "Nominal step cursor.", None),
        ("_pick_mode", "ModeTag", "Nominal mode tag.", None),
        ("_label", "LabelOption", "Nominal label option.", None),
        ("_filter_items", "ItemCutoff", "Nominal item cutoff.", "allows_item"),
    )
    picks = OwnerPicks(
        "picks.json",
        "",
        tuple(
            OwnerPick(
                finding=by_sym[sym].stable_id,
                owner=sym,
                because=f"{sym} owns its absence family",
                name=fam,
                judgement=doc,
                hoisted_name=hoist,
            )
            for sym, fam, doc, hoist in pick_specs
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 6
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    sim.apply()

    rewritten = (tmp_path / "pkg/compound.py").read_text(encoding="utf-8")
    ns_after: dict[str, object] = {}
    exec(compile(rewritten, "compound.py", "exec"), ns_after)
    assert ns_after["run_suite"]() == expected  # type: ignore[operator]

    _, _, after_findings = _scan(tmp_path)
    assert after_findings == []
