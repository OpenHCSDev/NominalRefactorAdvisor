"""End-to-end tests for Pattern 43 (`ZeroOneTupleToFamilyOperation`)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import codemod_plan_from_findings
from nominal_refactor_advisor.codemod_runtime import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import ZeroOneTupleDetector
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.scan_cache import ScanCache


def _import_fresh(root: Path, module_name: str):
    sys.path.insert(0, str(root))
    try:
        for name in [
            n
            for n in list(sys.modules)
            if n == module_name or n.startswith(f"{module_name}.")
        ]:
            del sys.modules[name]
        importlib.invalidate_caches()
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(root))


def _scan(root: Path):
    modules = parse_python_modules(root, use_parse_cache=False)
    snap = CodemodSourceSnapshot.from_modules(modules)
    findings = ZeroOneTupleDetector().detect(modules, DetectorConfig())
    return modules, snap, findings


def test_zero_one_tuple_pushes_for_loop_consumer_onto_family_without_iter(
    tmp_path: Path,
):
    source = (
        "from __future__ import annotations\n\n"
        "def maybe_tag(value: int) -> tuple[str, ...]:\n"
        "    if value <= 0:\n"
        "        return ()\n"
        "    return (f'v={value}',)\n\n"
        "def collect_tags(values: list[int]) -> list[str]:\n"
        "    out: list[str] = []\n"
        "    for v in values:\n"
        "        for tag in maybe_tag(v):\n"
        "            out.append(tag.upper())\n"
        "    return out\n"
    )
    mod_path = tmp_path / "tags.py"
    mod_path.write_text(source, encoding="utf-8")

    before_mod = _import_fresh(tmp_path, "tags")
    before_result = before_mod.collect_tags([-2, 0, 3, 7])
    assert before_result == ["V=3", "V=7"]

    with ScanCache.scope():
        _mods, snap, findings = _scan(tmp_path)
        assert len(findings) == 1

        unpicked = codemod_plan_from_findings(findings, selector_context=snap)
        rec = unpicked.report.records[0]
        assert rec.evaluation.recipe is None
        assert NeededInput.OWNER_PICK in rec.evaluation.needs
        assert "hoisted_name" in rec.evaluation.reason

        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=findings[0].stable_id,
                    owner="maybe_tag",
                    because="0/1 tuple presence belongs on a nominal tag match family",
                    name="TagMatch",
                    judgement="Nominal tag presence returned by maybe_tag.",
                    hoisted_name="record_into",
                    hoisted_judgement="Append this tag when present.",
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
        assert (
            replay.simulation.rewritten_sources[str(mod_path)]
            == sim.simulation.rewritten_sources[str(mod_path)]
        )
        sim.apply()

    after_text = mod_path.read_text(encoding="utf-8")
    assert "class TagMatch(ABC):" in after_text
    assert "class AbsentTagMatch(TagMatch):" in after_text
    assert "class HeldTagMatch(TagMatch):" in after_text
    assert "def __iter__" not in after_text
    assert "def record_into(" in after_text
    assert "maybe_tag(v).record_into(out)" in after_text
    assert "def maybe_tag(value: int) -> TagMatch:" in after_text

    after_mod = _import_fresh(tmp_path, "tags")
    assert after_mod.collect_tags([-2, 0, 3, 7]) == before_result

    with ScanCache.scope():
        _mods_after, _snap_after, findings_after = _scan(tmp_path)
        assert findings_after == []


def test_zero_one_tuple_iterable_consumers_cross_module_and_starred(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "producer.py").write_text(
        "from __future__ import annotations\n\n"
        "class Planner:\n"
        "    def _receipt(self, flag: bool, label: str) -> tuple[str, ...]:\n"
        "        if not flag:\n"
        "            return ()\n"
        "        return (label,)\n\n"
        "    def gather(self, flag: bool, label: str) -> tuple[str, ...]:\n"
        "        return ('head', *self._receipt(flag, label))\n\n"
        "    def list_upper(self, flag: bool, label: str) -> list[str]:\n"
        "        out = ['start']\n"
        "        out.extend(self._receipt(flag, label))\n"
        "        return [x.upper() for x in self._receipt(flag, label)] + out\n\n"
        "    def conditional(self, flag: bool, enable: bool, label: str) -> tuple[str, ...]:\n"
        "        return (*(self._receipt(flag, label) if enable else ()),)\n",
        encoding="utf-8",
    )
    (pkg / "consumer.py").write_text(
        "from __future__ import annotations\n\n"
        "from pkg.producer import Planner\n\n"
        "def stream_receipts(p: Planner, flag: bool, label: str):\n"
        "    yield from p._receipt(flag, label)\n",
        encoding="utf-8",
    )

    before_prod = _import_fresh(tmp_path, "pkg.producer")
    before_cons = _import_fresh(tmp_path, "pkg.consumer")
    p_before = before_prod.Planner()
    expected = (
        p_before.gather(True, "ok"),
        p_before.gather(False, "ok"),
        p_before.list_upper(True, "ok"),
        p_before.list_upper(False, "ok"),
        p_before.conditional(True, True, "ok"),
        p_before.conditional(True, False, "ok"),
        list(before_cons.stream_receipts(p_before, True, "ok")),
        list(before_cons.stream_receipts(p_before, False, "ok")),
    )
    assert expected == (
        ("head", "ok"),
        ("head",),
        ["OK", "start", "ok"],
        ["start"],
        ("ok",),
        (),
        ["ok"],
        [],
    )

    with ScanCache.scope():
        _mods, snap, findings = _scan(tmp_path)
        assert len(findings) == 1

        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=findings[0].stable_id,
                    owner="Planner._receipt",
                    because="planner receipt presence belongs on a nominal family",
                    name="ReceiptPresence",
                    judgement="Present or absent planner receipt.",
                ),
            ),
        )
        picked, report = picks.applied(findings)
        assert not report.unmatched
        plan = codemod_plan_from_findings(picked, selector_context=snap)
        assert plan.candidate_count == 1
        sim = plan.document.simulate(snap)
        assert sim.is_clean, sim.preflight_report
        sim.apply()

    after_prod_text = (pkg / "producer.py").read_text(encoding="utf-8")
    assert "class ReceiptPresence(ABC):" in after_prod_text
    assert "def __iter__(self) -> Iterator[str]:" in after_prod_text
    assert "return iter(())" in after_prod_text
    assert "yield self.value" in after_prod_text
    assert "return AbsentReceiptPresence()" in after_prod_text
    assert "return HeldReceiptPresence(label)" in after_prod_text

    after_prod = _import_fresh(tmp_path, "pkg.producer")
    after_cons = _import_fresh(tmp_path, "pkg.consumer")
    p_after = after_prod.Planner()
    actual = (
        p_after.gather(True, "ok"),
        p_after.gather(False, "ok"),
        p_after.list_upper(True, "ok"),
        p_after.list_upper(False, "ok"),
        p_after.conditional(True, True, "ok"),
        p_after.conditional(True, False, "ok"),
        list(after_cons.stream_receipts(p_after, True, "ok")),
        list(after_cons.stream_receipts(p_after, False, "ok")),
    )
    assert actual == expected

    with ScanCache.scope():
        _mods_after, _snap_after, findings_after = _scan(tmp_path)
        assert findings_after == []


def test_zero_one_tuple_refuses_non_iterable_consumers_overrides_and_unclosed_public(
    tmp_path: Path,
):
    (tmp_path / "bad.py").write_text(
        "from __future__ import annotations\n\n"
        "def _by_index(flag: bool) -> tuple[int, ...]:\n"
        "    if flag:\n"
        "        return (1,)\n"
        "    return ()\n\n"
        "def _uses_index(flag: bool) -> int:\n"
        "    return _by_index(flag)[0]\n\n"
        "def _by_truthiness(flag: bool) -> tuple[int, ...]:\n"
        "    if flag:\n"
        "        return (1,)\n"
        "    return ()\n\n"
        "def _uses_truth(flag: bool) -> bool:\n"
        "    return bool(_by_truthiness(flag))\n\n"
        "def _by_assign(flag: bool) -> tuple[int, ...]:\n"
        "    if flag:\n"
        "        return (1,)\n"
        "    return ()\n\n"
        "def _uses_assign(flag: bool) -> tuple[int, ...]:\n"
        "    val = _by_assign(flag)\n"
        "    return val\n\n"
        "def public_no_callers(flag: bool) -> tuple[int, ...]:\n"
        "    if flag:\n"
        "        return (1,)\n"
        "    return ()\n\n"
        "class Base:\n"
        "    def item(self, flag: bool) -> tuple[int, ...]:\n"
        "        if flag:\n"
        "            return (1,)\n"
        "        return ()\n\n"
        "class Sub(Base):\n"
        "    def item(self, flag: bool) -> tuple[int, ...]:\n"
        "        return ()\n\n"
        "def _use_override(b: Base, flag: bool) -> list[int]:\n"
        "    return list(b.item(flag))\n",
        encoding="utf-8",
    )
    with ScanCache.scope():
        _mods, snap, findings = _scan(tmp_path)
        plan = codemod_plan_from_findings(findings, selector_context=snap)
        by_symbol = {
            rec.finding.primary_evidence.subject_symbol: rec.evaluation
            for rec in plan.report.records
        }
        assert "_by_index" in by_symbol
        assert "non-iterable context" in by_symbol["_by_index"].reason
        assert "_by_truthiness" in by_symbol
        assert "non-iterable context" in by_symbol["_by_truthiness"].reason
        assert "_by_assign" in by_symbol
        assert "non-iterable context" in by_symbol["_by_assign"].reason
        assert "public_no_callers" in by_symbol
        assert "no scanned callers" in by_symbol["public_no_callers"].reason
        assert "Base.item" in by_symbol
        assert "overriding hierarchy" in by_symbol["Base.item"].reason
        for ev in by_symbol.values():
            assert ev.recipe is None
            assert NeededInput.OWNER_PICK not in ev.needs
