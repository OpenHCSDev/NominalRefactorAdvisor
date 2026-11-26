"""Tests for pushing scanned-class foreign absence branches onto their owner class."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodPlanSequence
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import (
    OwnStateBranchAnswerDetector,
)
from nominal_refactor_advisor.detectors._restated_facts import (
    ForeignAbsenceProbeDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPicks
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
        findings = ForeignAbsenceProbeDetector().detect(modules, DetectorConfig())
    return modules, snap, findings


def _apply_one(finding, snap: CodemodSourceSnapshot) -> None:
    evaluation = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
        finding, snap
    )
    assert evaluation.rejection_reason == ""
    sim = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snap)
    assert sim.is_clean
    sim.apply()


def test_whole_function_early_return_and_guard_push_cross_module_with_parity(
    tmp_path: Path,
) -> None:
    """Whole-function early-return and compound `and` guard probes on scanned classes
    synthesize directly as executable recipes, push onto the owner class, clear the
    finding, introduce no `OWN_STATE_BRANCH_ANSWER`, and preserve runtime parity."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/models.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Budget:\n"
                "    token_limit: int | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Snapshot:\n"
                "    attempt: int | None\n"
            ),
            "pkg/service.py": (
                "from pkg.models import Budget, Snapshot\n\n\n"
                "def char_ceiling(budget: Budget, chars_per_token: int) -> int:\n"
                "    if budget.token_limit is None:\n"
                "        return 0\n"
                "    return budget.token_limit * chars_per_token\n\n\n"
                "def require_deferral(snapshot: Snapshot, authorized: bool) -> None:\n"
                "    if snapshot.attempt is not None and not authorized:\n"
                "        raise ValueError('unauthorized')\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_symbol = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_symbol) == {"char_ceiling", "require_deferral"}

    for sym in ("char_ceiling", "require_deferral"):
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        _apply_one(finding, snap)

    after_models = (tmp_path / "pkg/models.py").read_text()
    after_service = (tmp_path / "pkg/service.py").read_text()
    assert "def char_ceiling(self, chars_per_token):" in after_models
    assert "def require_deferral(self, authorized):" in after_models
    assert "return budget.char_ceiling(chars_per_token)" in after_service
    assert "snapshot.require_deferral(authorized)" in after_service

    # Re-scan: zero foreign absence probes and zero own-state branch answers.
    after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []
    with ScanCache.scope():
        assert OwnStateBranchAnswerDetector().detect(after_mods, DetectorConfig()) == []

    # Runtime execution parity.
    sys.path.insert(0, str(tmp_path))
    try:
        for mod_name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
            del sys.modules[mod_name]
        import pkg.models as models  # type: ignore[import-not-found]
        import pkg.service as service  # type: ignore[import-not-found]

        assert service.char_ceiling(models.Budget(None), 4) == 0
        assert service.char_ceiling(models.Budget(10), 4) == 40
        service.require_deferral(models.Snapshot(None), False)
        service.require_deferral(models.Snapshot(1), True)
        try:
            service.require_deferral(models.Snapshot(1), False)
        except ValueError as err:
            assert "unauthorized" in str(err)
        else:
            raise AssertionError("expected ValueError")
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in [m for m in sys.modules if m == "pkg" or m.startswith("pkg.")]:
            del sys.modules[mod_name]


def test_embedded_void_ternary_and_loop_continue_use_owner_picks_with_parity(
    tmp_path: Path,
) -> None:
    """Embedded void `if`, non-boolean ternary `IfExp`, and loop `continue` probes stop at
    `NeededInput.OWNER_PICK` and apply cleanly once `--codemod-owner-picks` supplies names."""

    _write(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/work.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Event:\n"
                "    payload: str | None\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Row:\n"
                "    sequence: int | None\n\n\n"
                "def record_event(event: Event, sink: list[str]) -> int:\n"
                "    before = len(sink)\n"
                "    if event.payload is not None:\n"
                "        sink.append(event.payload)\n"
                "    return len(sink) - before\n\n\n"
                "def classify_row(row: Row) -> str:\n"
                "    label = 'missing' if row.sequence is None else f'seq:{row.sequence}'\n"
                "    return label\n\n\n"
                "def collect_present(events: list[Event]) -> list[str]:\n"
                "    out: list[str] = []\n"
                "    for event in events:\n"
                "        if event.payload is None:\n"
                "            continue\n"
                "        out.append(event.payload.upper())\n"
                "    return out\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {"record_event", "classify_row", "collect_present"}

    # Without picks, all three stop cleanly at OWNER_PICK.
    for sym in ("record_event", "classify_row", "collect_present"):
        ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(
            by_sym[sym], snap
        )
        assert ev.needs == (NeededInput.OWNER_PICK,), (sym, ev.rejection_reason, ev.needs)

    # Apply each with an OwnerPicks entry.
    pick_specs = {
        "record_event": ("append_payload", "Append payload to sink when present."),
        "classify_row": ("sequence_label", "Format sequence label."),
        "collect_present": ("append_upper_payload", "Append uppercased payload when present."),
    }
    for sym, (name, judgement) in pick_specs.items():
        _mods, snap, cur = _scan(tmp_path)
        finding = next(f for f in cur if f.primary_evidence.symbol == sym)
        picks_path = tmp_path / f"{sym}.picks.json"
        picks_path.write_text(
            json.dumps(
                {
                    "picks": [
                        {
                            "finding": finding.stable_id,
                            "owner": sym,
                            "because": "test pick",
                            "name": name,
                            "judgement": judgement,
                        }
                    ]
                }
            )
        )
        picked_findings, report = OwnerPicks.read(picks_path).applied(cur)
        assert not report.unmatched
        picked = next(f for f in picked_findings if f.primary_evidence.symbol == sym)
        _apply_one(picked, snap)

    after_source = (tmp_path / "app/work.py").read_text()
    assert "def append_payload(self, sink):" in after_source
    assert "def sequence_label(self):" in after_source
    assert "def append_upper_payload(self, out):" in after_source
    assert "event.append_payload(sink)" in after_source
    assert "row.sequence_label()" in after_source
    assert "event.append_upper_payload(out)" in after_source

    _after_mods, _after_snap, remaining = _scan(tmp_path)
    assert remaining == []

    ns: dict[str, object] = {}
    exec(compile(after_source, str(tmp_path / "app/work.py"), "exec"), ns)
    Event = ns["Event"]
    Row = ns["Row"]
    sink: list[str] = []
    assert ns["record_event"](Event(None), sink) == 0  # type: ignore[operator]
    assert ns["record_event"](Event("hi"), sink) == 1  # type: ignore[operator]
    assert sink == ["hi"]
    assert ns["classify_row"](Row(None)) == "missing"  # type: ignore[operator]
    assert ns["classify_row"](Row(7)) == "seq:7"  # type: ignore[operator]
    assert ns["collect_present"]([Event(None), Event("a"), Event("b")]) == ["A", "B"]  # type: ignore[operator]


def test_soundness_refusals_for_bool_predicates_escaping_locals_protocols_and_collisions(
    tmp_path: Path,
) -> None:
    """Refuses bare boolean predicates, escaping outer local assignments, Protocol owners,
    and method-name collisions without producing dirty edits."""

    _write(
        tmp_path,
        {
            "bad/__init__.py": "",
            "bad/mod.py": (
                "from dataclasses import dataclass\n"
                "from typing import Protocol\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    value: int | None\n\n"
                "    def existing(self) -> int:\n"
                "        return 1\n\n\n"
                "class BoxProto(Protocol):\n"
                "    value: int | None\n\n\n"
                "def is_filled(box: Box) -> bool:\n"
                "    return box.value is not None\n\n\n"
                "def bool_ternary(box: Box) -> bool:\n"
                "    return True if box.value is not None else False\n\n\n"
                "def mutates_outer_local(box: Box) -> int:\n"
                "    total = 0\n"
                "    if box.value is not None:\n"
                "        total = box.value\n"
                "    return total\n\n\n"
                "def proto_consumer(box: BoxProto) -> int:\n"
                "    if box.value is None:\n"
                "        return 0\n"
                "    return box.value\n\n\n"
                "def existing(box: Box) -> int:\n"
                "    if box.value is None:\n"
                "        return 0\n"
                "    return box.value\n\n\n"
                "def nullable_subject_guard(box: Box | None) -> int:\n"
                "    if box is None or box.value is None:\n"
                "        return 0\n"
                "    return box.value\n\n\n"
                "def short_circuit_prefix_guard(ready: bool, box: Box | None) -> None:\n"
                "    if ready and box.value is not None:\n"
                "        raise ValueError('boom')\n"
            ),
        },
    )

    _mods, snap, findings = _scan(tmp_path)
    by_sym = {f.primary_evidence.symbol: f for f in findings}
    assert set(by_sym) == {
        "is_filled",
        "bool_ternary",
        "mutates_outer_local",
        "proto_consumer",
        "existing",
        "nullable_subject_guard",
        "short_circuit_prefix_guard",
    }

    det = ForeignAbsenceProbeDetector()
    for sym in (
        "is_filled",
        "bool_ternary",
        "mutates_outer_local",
        "proto_consumer",
        "nullable_subject_guard",
        "short_circuit_prefix_guard",
    ):
        ev = det.evaluate_recipe_for_finding(by_sym[sym], snap)
        assert ev.rejection_reason != "", f"{sym} must be refused"
        assert NeededInput.OWNER_PICK not in ev.needs, (sym, ev.rejection_reason)

    collision_ev = det.evaluate_recipe_for_finding(by_sym["existing"], snap)
    assert "already declares `existing`" in collision_ev.rejection_reason
    assert collision_ev.needs == (NeededInput.OWNER_PICK,)


def test_cross_module_free_global_constant_is_refused(tmp_path: Path) -> None:
    """Moving a branch across modules that reads an unproved module-level constant
    must be refused instead of dropping the name and raising NameError at runtime."""

    _write(
        tmp_path,
        {
            "cross/__init__.py": "",
            "cross/owner.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Item:\n"
                "    cached: str | None\n"
            ),
            "cross/consumer.py": (
                "from cross.owner import Item\n\n\n"
                "_SENTINEL = '__cached__'\n\n\n"
                "def read_cached(item: Item) -> str:\n"
                "    if item.cached is not None:\n"
                "        return item.cached\n"
                "    return _SENTINEL\n"
            ),
        },
    )
    _mods, snap, findings = _scan(tmp_path)
    (finding,) = findings
    ev = ForeignAbsenceProbeDetector().evaluate_recipe_for_finding(finding, snap)
    assert "_SENTINEL" in ev.rejection_reason
    assert ev.needs == (NeededInput.SOURCE_CHANGE,)

